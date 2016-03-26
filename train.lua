
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

-- require('mobdebug').start()

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.ReadSlice'
require 'util.WriteSlice'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'
local CGRU = require 'model.CGRU'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'cgru', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',2,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',10000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-cgru_width',15,'width of the cgru "mental image"')
cmd:option('-cgru_height',15,'height of the cgru "mental image"')
cmd:option('-cgru_in_x', 1, 'position x to write to in mental image')
cmd:option('-cgru_in_y', 1, 'position y to write to in mental image')
cmd:option('-cgru_out_x', 0, 'position x to read from in mental image (0 = cgru_in_x)')
cmd:option('-cgru_out_y', 0, 'position y to read from in mental image (0 = cgru_in_y)')
cmd:option('-grad_noise', 1, 'whether to use grad_noise. hardcoded at stddev = 1/(iteration ^ (1/4)); *= grad:mean()')
cmd:option('-hardmode', 0, 'whether to use hardmode. if 1, use "percentage of correct top-1s" as the displayed loss (NLL still used to train).')
cmd:text()
-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- TODO: bayesian optimization - might be that we can wrap the thingy?
-- TODO: spearmint looks pretty easy to use. promising! if we can't get things
-- TODO: to work, then hook up to spearmint. The CGRU paper used grid search,
-- TODO: and couldn't get good results without it.
-- TODO: (later) oh, I already made this note. I guess I've made it in two
-- TODO: places now.

if opt.model == 'cgru' then
    -- we don't even try to have more than this at the moment.
    -- TODO: but it might be a good idea to let it. the problem is, do we really
    --       want to have a bunch of layers with permanently unshared weights?
    --       I think what we really want is to repeat the same layer, but then
    --       also add parameter sharing relaxation. maybe an option for number
    --       of iterations to repeatedly invoke the network with the same input?
    --       or with no input. though probably that'll completely fall flat with
    --       RNNs, because they have no memory to speak of. the CGRU might
    --       benefit, but without parameter sharing relaxation it'll run into
    --       the same old problem.
    opt.num_layers = 1
end
if opt.cgru_out_x == 0 then
    opt.cgru_out_x = opt.cgru_in_x
end
if opt.cgru_out_y == 0 then
    opt.cgru_out_y = opt.cgru_in_y
end

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

function asgpu(v)
    if opt.gpuid >=0 and opt.opencl == 0 then
        return v:float():cuda()
    end
    if opt.gpuid >=0 and opt.opencl == 1 then
        return v:cl()
    end
end


-- initialize cunn/cutorch for training on the GPU and bail and tell the user
-- if no gpu is available
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    require 'cudnn'
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        os.exit(1)
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    local checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i) then
            vocab_compatible = false
        end
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    end
    if not (checkpoint_vocab_size == vocab_size) then
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'cgru' then
        protos.rnn = CGRU.cgru(vocab_size, opt.rnn_size, opt.dropout,
                            opt.cgru_in_x, opt.cgru_in_y,
                            opt.cgru_out_x, opt.cgru_out_y)
    end
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
if opt.model == 'cgru' then
    h_init = asgpu(torch.zeros(opt.batch_size, opt.rnn_size, opt.cgru_width, opt.cgru_height))
    init_state[1] = h_init
else
    for L=1,opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        h_init = asgpu(h_init)
        table.insert(init_state, h_init:clone())
        if opt.model == 'lstm' then
            table.insert(init_state, h_init:clone())
        end
    end
end

-- ship the model to the GPU if desired
for k,v in pairs(protos) do asgpu(v) end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    -- TODO: what initialization did n-gpu/c-gru paper use?
    -- TODO: initializing to weights of 1 seems pretty likely to be a huge win
    --       even for cgru, because cgru still needs to remember things. cgru
    --       is more like a crapton of grus all feeding into each other; it still
    --       has the memory problem. that said, the gru arch does the same job,
    --       just slower. TODO: do we need that speed?
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    x = asgpu(x)
    y = asgpu(y)
    return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch

        -- return: tensor(batch_size, seq_length), tensor(batch_size, seq_length)
        local x, y = loader:next_batch(split_index)

        -- return: tensor(seq_length, batch_size), tensor(seq_length, batch_size)
        x,y = prepro(x,y)

        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            -- x = tensor(seq_length, batch_size) so
            -- x[t] = tensor(batch_size)
            -- yes, that's one dimension! it's an array of char indexes.

            -- rnn_state[t-1][*] = tensor(batch_size, ...)
            -- cgru: ... = opt.rnn_size (ie, m), opt.cgru_width, opt.cgru_height
            -- rnn-like: ... = opt.rnn_size

            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}

            -- this assumes that the final output is the prediction
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 

            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- carry over state between batches
        -- this depends on batch continuity being established in the loader;
        -- the particular series of transforms gives us that guarantee.
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
grad_noise = torch.Tensor()
grad_noise = asgpu(grad_noise)
grad_noise:resizeAs(grad_params)

function hardloss(batch_predictions, batch_ys)
    local _, maxes = batch_predictions:max(2)
    local b = batch_ys:clone()
    b:add(50)
    maxes:add(50)
    b:cdiv(asgpu(maxes:double()))
    b:add(-1)
    b:sign()
    b:abs()
    b:add(-1)
    b:mul(-1)
    return b:mean()
end

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        local curloss = clones.criterion[t]:forward(predictions[t], y[t])
        if opt.hardmode == 1 then 
            loss = loss + hardloss(predictions[t], y[t])
        else
            loss = loss + curloss
        end
    end
    -- TODO: parameter sharing relaxation - not just for cgru
    -- TODO: if opt.param_sharing_relaxation then
    -- TODO:     ??????
    -- TODO:     something regarding unsharing the weights of clones and then
    -- TODO:     mathing it somehow? need to actually read the end of the paper,
    -- TODO:     just skimmed it. how do they decide which instance to use? are
    -- TODO:     there just a fixed number and that's the max number of steps?
    -- TODO:     or are they doing something like round robin, or is the network
    -- TODO:     deciding which next layer to run? (also, how did that paper
    -- TODO:     that does submodel selection with q learners do it?)
    -- TODO:
    -- TODO:     see the end of "neural gpus learn algorithms" and I don't
    -- TODO:     remember the name of the q learning paper in question
    -- TODO:     ??????
    -- TODO: end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    -- TODO: gradient noise - probably already solved in torch, or trivially easy
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- NOTE: I don't think this needs to be a clone, right?
    init_state_global = rnn_state[#rnn_state]

    -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- grad_params:div(opt.seq_length)

    -- clip gradient element-wise
    
    if opt.grad_noise then
        grad_noise:mul(grad_params:mean() * opt.grad_noise)
        grad_params:add(grad_noise)
    end
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    if opt.grad_noise then
        grad_noise:normal(0, 1.0/(i^(1/4)))
    end
    local _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real
    
    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 1 then
        local gradnorm = grad_params:norm()
        local paramnorm = params:norm()
        print(string.format("%d/%d (epoch %.3f), train_loss = %.8g, grad norm/std/mean = %.4g/%.4g/%.4g, param n/s/m = %.4g/%.4g/%.4g, grad/param norm = %.4g, time: batch = %.4fs, unroll = %.4f/s", i, iterations, epoch, train_loss,
                gradnorm, grad_params:std(), grad_params:mean(),
                paramnorm, params:std(), params:mean(),
                gradnorm / paramnorm,
                time, opt.batch_size / time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


