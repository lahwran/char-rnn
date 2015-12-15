

local CGRU = {}

--[[
Creates one timestep of one Convolutional Gated Recurrent Unit
(or "neural gpu"; which, amusingly, is a bad approximation of mental neurons
and also a bad approximation of graphics processing units)

Paper reference: http://arxiv.org/abs/1511.08228

]]--

-- in the original paper, the whole sequence is fed at once. in this
-- implementation we're currently passing it one item at a time. This might not
-- be a good idea; it'd be good to:
-- 1. TODO: make passing input optional. if we just want to think for a bit,
--          we should be able to pass nulls. it's up to the training system to
--          decide this.
-- 2. TODO: make input variable-size. if we pass multiple inputs in one go,
--          replace more cells.
--
-- h/prev_h: hidden state, ie memory
-- spearmint - not state of the art but oh well (as usual for public
--          impls, when will the world learn to publish code...) probably could
--          benefit a lot from a gpu cluster though. parameter search on ec2?
--          -> expensive, just let it run
-- hyperparameter for network width
-- hyperparameter for network height
-- hyperparameter for m embedding size, when using LookupTable
-- hyperparameter for context amount?
-- hyperparameter for cell into which input is written at each timestep
-- hyperparameter for cell from which output is read at each timestep
-- hyperparameter boolean to ignore "output location" hyperparameter and
--          use the same as the input
-- hyperparameter for convolution size x/y
-- hyperparameter for having hyperparameters
-- hyperparameter for how many TODOs to write
--
-- TODO: probably want to do rolling unrolling for many more timesteps now
-- TODO: segmented data
-- TODO: per-line prediction is actually viable with this, though it'll need thinking time
function CGRU.cgru(input_size, m, dropout, in_x, in_y, out_x, out_y)
    dropout = dropout or 0 
    -- there is only one input
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- character index
    table.insert(inputs, nn.Identity()()) -- prev_h
    
    embedded = nn.LookupTable(input_size, m)()
    local updated_h = WriteSlice({{}, in_x, in_y})({embedded, inputs[1]})

    local prev_h = updated_h


    function conv(activation, m, hidden)
        local kernsize = 3 -- original paper only uses 3x3 convolutions.
        local stride = 1 -- leave this at 1
        local padding = 1 -- w = (w + 2*p - k) + 1 solve for p where k = 3
                          -- (ie (k-1)/2)
        return activation(nn.SpatialConvolution(m, m, kernsize, kernsize,
                                            stride, stride,
                                            padding, padding)(hidden))
    end

    -- TODO: for each dimension that is replaced in each forward step, the
    -- TODO: backward step needs to provide gradients. the obvious thing to do
    -- TODO: is to erase the gradients at that location and replcae them with 0.
    -- TODO: this is likely best solved via a custom module.
    --
    -- TODO: the model will likely learn better rules if the whole first column
    -- TODO: is replaced at every time step with a history. test this.

    local outputs = {}
    -- TODO: dropout: x = nn.Dropout(dropout)(x) needs to happen at the end
    --
    -- TODO: OneHot(input_size) needs to happen outside now. preprocess step?
    -- TODO: alternately, could just use nn.LookupTable, which goes straight
    -- TODO: from indexes to learnable embeddings, and takes an array of indexes.

    -- GRU tick
    -- forward the update and reset gates
    local update_gate = conv(nn.Sigmoid(), m, prev_h)
    local reset_gate = conv(nn.Sigmoid(), m, prev_h)
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local hidden_candidate = conv(nn.Tanh(), m, gated_hidden)
    -- compute new interpolated hidden state, based on the update gate
    local squashed_new_hidden = nn.CMulTable()({update_gate, hidden_candidate})

    -- TODO: why x * -1 + 1? what's wrong with 1 - x? silly computers, equal
    --      math should be equally fast and good.
    local derp = nn.MulConstant(-1,false)(update_gate)
    local inverted_update_gate = nn.AddConstant(1,false)(derp)

    local squashed_prev_h = nn.CMulTable()({inverted_update_gate, prev_h})
    local next_h = nn.CAddTable()({squashed_prev_h, squashed_new_hidden})
    if dropout > 0 then next_h = nn.Dropout(dropout)(next_h) end

    table.insert(outputs, next_h)
    -- set up the decoder
    -- TODO: need to slice out just the output field. would be good to make this
    -- TODO: the column that will be overwritten at each time step, so that
    -- TODO: we still get gradients on it - loss gradients on the output column,
    -- TODO: and next-time-step gradients on the rest of memory. How do we deal
    -- TODO: with the final layer, though? zero gradients for everything else,
    -- TODO: probably?
    --

    local sliced = ReadSlice({{}, out_x, out_y})(next_h)
    local proj = nn.Linear(m, input_size)(sliced)
    local logsoft = nn.LogSoftMax()(proj)
    -- TODO: the small range of gradients means we will only get gradients for
    -- TODO: the immediately adjacent columns in the first timestep, and
    -- TODO: one step out from those in the one before that, and so on. This
    -- TODO: means that we're going to get a lot of filters that are nothing but
    -- TODO: "move" operators, probably? because it'll be advantageous to do that.
    -- TODO: might be worth setting a sparsity rule, too, in addition to dropout,
    -- TODO: but maybe not.
    --
    -- TODO: Also, it might be worth letting it think for more
    -- TODO: time steps in between giving it a next character, because it's
    -- TODO: not very much time to predict the next character every time. It's
    -- TODO: not too horrible, though, because every character is much more often
    -- TODO: than every word.
    table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

return CGRU
