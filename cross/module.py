import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class ReduceLRonPlateau():
    def __init__(self, optimizer, factor=0.1, patience=5, cooldown=10, min_delta=0.0001):
        self.lr = optimizer.lr.numpy()
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.wait = 0   # Patience counter
        self.cooldown_counter = 0
        self.cooldown = cooldown
        self.best = 1e9 # Best metric to compare against
        self.min_delta = min_delta

    def __call__(self, current):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        if current - self.min_delta < self.best:
            self.best = current
            self.wait = 0
        elif self.cooldown_counter == 0:
            self.wait += 1
            if self.wait >= self.patience:
                self.optimizer.lr.assign(self.optimizer.lr.numpy() * self.factor)
                self.cooldown_counter = self.cooldown
                self.wait = 0
        
        return self.cooldown_counter > 0

    def get_lr(self):
        return self.optimizer.lr.numpy()

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.wait = 0   # Patience counter
        self.min_delta = min_delta
        self.best = 1e9

    def __call__(self, current):
        if current - self.min_delta < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        
        return False

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask=None):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.math.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        if mask is not None:
            score = score + (mask * -1e9)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)

    def call(self, query, values, mask=None):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # mask shape == (batch_size, max_len)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, 1, max_len)
        score = tf.matmul(query_with_time_axis, self.W(values), transpose_b=True)
        if mask is not None:
            score = score + (mask * -1e9)

        # attention_weights shape == (batch_size, 1, max_len)
        attention_weights = tf.nn.softmax(score, axis=2)

        # context_vector shape after sum == (batch_size, value_size)
        context_vector = tf.matmul(attention_weights, values)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class StructuralLuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(StructuralLuongAttention, self).__init__()
        self.v = tf.keras.layers.Dense(1)

        self.W_h = tf.keras.layers.Dense(units)
        self.W_v = tf.keras.layers.Dense(units)
        
        # Positional bias
        self.W_p = tf.keras.layers.Dense(units)
        self.timestep = 1

        # Markov assumption
        self.W_m = tf.keras.layers.Dense(units)
        self.previous_scores = None

    def reset(self):
        self.timestep = 1
        self.previous_scores = None

    def call(self, query, values, mask):
        # query hidden state shape == (batch_size, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # mask shape == (batch_size, max_len, 1)
        max_len = mask.shape[1]
        if self.timestep == 1:
            self.previous_scores = tf.zeros(mask.shape)

        # we are doing this to broadcast addition along the time axis to calculate the score
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # input_lengths shape == (batch_size, max_len, 1)
        input_lengths = tf.cast(tf.math.equal(mask, 0), tf.float32) # (batch_size, max_len, 1)
        input_lengths = 1. + tf.reduce_sum(input_lengths, axis=1) # (batch_size, 1)
        input_lengths = tf.tile(input_lengths, tf.constant([1, max_len])) # (batch_size, max_len)
        input_lengths = tf.expand_dims(input_lengths, 2)

        # positional shape == (batch_size, max_len, 2)
        positional = tf.convert_to_tensor([1 + self.timestep for _ in range(max_len)]) # (max_len,)
        positional = tf.stack([positional, 2 + tf.range(max_len)]) # (2, max_len)
        positional = tf.cast(tf.transpose(positional), tf.float32) # (max_len, 2)
        positional = tf.expand_dims(positional, 0)
        positional = tf.tile(positional, tf.constant([mask.shape[0], 1, 1]))

        # positional shape == (batch_size, max_len, 3)
        positional = tf.concat([positional, input_lengths], axis=-1)
        positional = tf.math.log(positional)

        # markov shape == (batch_size, max_len, 5)
        markov = tf.pad(self.previous_scores, tf.constant([[0, 0], [2, 2], [0, 0]])) # (batch_size, max_len+4, 1)
        markov = tf.stack([markov[:, i:i+5, 0] for i in range(max_len)], axis=1)

        # summand shape == (batch_size, max_len, units)
        summand = self.W_h(query_with_time_axis) + self.W_v(values) + \
                  self.W_p(positional) + self.W_m(markov)
        
        # score shape == (batch_size, max_len, 1)
        score = self.v(tf.math.tanh(summand))
        score = score + (mask * -1e9)

        # attention_weights shape == (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, value_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        self.timestep += 1
        self.previous_scores = attention_weights

        return context_vector, attention_weights

def create_padding_mask(seq, mode='luong'):
    seq = tf.math.equal(seq, 0)
    if mode == 'transformer':
        seq = tf.cast(seq, tf.float32)
        return tf.reshape(seq, (seq.shape[0], 1, 1, seq.shape[1])) # (batch_size, 1, 1, seq_len)
    elif mode == 'luong':
        seq = tf.cast(seq, tf.float32)
        return tf.expand_dims(seq, 1) # (batch_size, 1, seq_len)
    elif mode == 'structure' or mode == 'bahdanau':
        seq = tf.cast(seq, tf.float32)
        return tf.expand_dims(seq, 2) # (batch_size, seq_len, 1)
    elif mode == 'lstm':
        seq = tf.math.logical_not(seq)
        return seq

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
      to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class SingleHeadTransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, rate=0.2):
        super(SingleHeadTransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, 1)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        return out1

class TagEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
           cnst_tag=False, rate=0.2):
        super(TagEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        if num_layers == 1 and num_heads == 1:
            self.enc_layers = [SingleHeadTransformerEncoderLayer(d_model, rate)]
        else:
            self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.cnst_tag = cnst_tag
        if self.cnst_tag:
            # hidden size is (d_model // num_heads)
            self.tag_attention = LuongAttention(2 * d_model // num_heads)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def attend(self, enc_vec, tag_vecs, tag_mask=None):
        assert self.cnst_tag

        enc_vec = enc_vec[:, -1, :]
        tag_context_vector, tag_attention_weights = self.tag_attention(enc_vec, tag_vecs, tag_mask)
        return tag_context_vector, tag_attention_weights

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, rate=0.2):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                       self.enc_units,
                                       return_sequences=True,
                                       return_state=True))
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, state=None, mask=None, training=True):
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        output, *state = self.lstm(x, initial_state=state, mask=mask)
        
        state_h = tf.concat([state[0], state[2]], axis=-1)
        state_c = tf.concat([state[1], state[3]], axis=-1)
        
        return output, state_h, state_c

    def initial_state(self, batch_sz):
        return [tf.zeros((batch_sz, self.enc_units)) for _ in range(4)]

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, inc_tags=False, 
                 use_ptv=False, cnst_tag=False, rate=0.2):
        super(Decoder, self).__init__()
        self.dec_units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(2*self.dec_units,
                                       return_sequences=True,
                                       return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.inc_tags = inc_tags
        self.use_ptv = use_ptv
        self.cnst_tag = cnst_tag

        if cnst_tag:
            # used for attention
            self.enc_attention = StructuralLuongAttention(self.dec_units)
            self.dropout3 = tf.keras.layers.Dropout(rate)
        if inc_tags:
            # used for attention
            self.tag_attention = LuongAttention(2*self.dec_units)
            self.enc_attention = StructuralLuongAttention(self.dec_units)
            self.dropout3 = tf.keras.layers.Dropout(rate)
        else:
            self.enc_attention = StructuralLuongAttention(self.dec_units)

        if self.use_ptv:
            self.W_ptv = tf.keras.layers.Dense(2*self.dec_units)
            
        self.W_c = tf.keras.layers.Dense(2*self.dec_units)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def reset(self):
        self.enc_attention.reset()

    def call(self, x, state, enc_output, tag_vecs=None, enc_mask=None,
             tag_mask=None, training=True, ptv=None):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        if self.cnst_tag:
            assert tag_vecs is not None

            s_prev, state_c = state
            
            s_k = tf.concat([s_prev, tag_vecs], axis=-1)
            s_k = tf.math.tanh(self.dropout3(self.W_c(s_k)))
            
            # Attend over encoder vectors
            enc_context_vector, attention_weights = self.enc_attention(s_k, enc_output, enc_mask)

            state_c = enc_context_vector
            output, state_h, state_c = self.lstm(x, initial_state=(s_prev, state_c))

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))
            output = self.dropout2(output)

            attention_output = attention_weights
            state = (state_h, state_c)
        elif self.inc_tags:
            assert tag_vecs is not None

            s_prev, state_c = state
            # Attend over tag vectors
            tag_context_vector, tag_attention_weights = self.tag_attention(s_prev, tag_vecs, tag_mask)
            
            s_k = tf.concat([s_prev, tag_context_vector], axis=-1)
            s_k = tf.math.tanh(self.dropout3(self.W_c(s_k)))
            
            # Attend over encoder vectors
            enc_context_vector, attention_weights = self.enc_attention(s_k, enc_output, enc_mask)

            state_c = enc_context_vector
            output, state_h, state_c = self.lstm(x, initial_state=(s_prev, state_c))

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))
            output = self.dropout2(output)

            attention_output = (attention_weights, tag_attention_weights)
            state = (state_h, state_c)
        else:
            # passing the concatenated vector to the LSTM
            output, state_h, state_c = self.lstm(x, initial_state=state)

            # output shape == (batch_size * 1, 2*hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))
            enc_context_vector, attention_weights = self.enc_attention(state_h, enc_output, enc_mask)

            # output shape == (batch_size, 3*hidden_size)
            output = tf.concat([output, enc_context_vector], axis=-1)

            # output shape == (batch_size, 2*hidden_size)
            output = tf.math.tanh(self.dropout2(self.W_c(output)))
            
            attention_output = attention_weights
            state = (state_h, state_c)

        if self.use_ptv:
            output = tf.concat([output, ptv], axis=-1)
            output = tf.math.tanh(self.dropout2(self.W_ptv(output)))

        # output shape == (batch_size, vocab)
        x = self.fc(output)
        x = self.dropout1(x, training=training)

        return x, state, attention_output
