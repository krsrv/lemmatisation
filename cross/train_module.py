import tensorflow as tf
from module import create_padding_mask

import random

class TrainStep():
    def __init__(self, char_encoder, tag_encoder, decoder):
        self.char_encoder = char_encoder
        self.tag_encoder = tag_encoder
        self.decoder = decoder

        self.random = tf.random.uniform
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True, reduction='none')

        self.____call__tf = tf.function(self.____call__)

    def reset(self):
        self.____call__tf = None
        self.____call__tf = tf.function(self.____call__)

    # @tf.function
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Original function call to train_step
    def __call__(self, batch_dataset, char_embedder, tag_embedder, dense_fc, start_token, 
                 optimizer, mask=0, mode='P1', training=True, return_outputs=False, 
                 return_attention_plots=False):
        inputs, targets, tags = batch_dataset
        batch_size = inputs.shape[0]

        count = tf.cast(tf.ones((batch_size,)), tf.bool)
        outputs = tf.zeros((batch_size, 1), tf.int32)
        char_attention_plot = tf.zeros((batch_size, 1, inputs.shape[1]))
        tag_attention_plot = tf.zeros((batch_size, 1, tags.shape[1]))
        dec_input = tf.ones((batch_size, 1), tf.int32) * start_token
        loss = 0

        if True:
            return self.____call__tf(inputs, targets, tags, char_embedder, tag_embedder, dense_fc, 
                 start_token, optimizer, mask, mode, training, return_outputs,
                 return_attention_plots, count, outputs, char_attention_plot, tag_attention_plot,
                 dec_input, loss)
        else:
            return self.____call__(inputs, targets, tags, char_embedder, tag_embedder, dense_fc, 
                 start_token, optimizer, mask, mode, training, return_outputs,
                 return_attention_plots, count, outputs, char_attention_plot, tag_attention_plot,
                 dec_input, loss)

    # Variables pushed out of original function to allow decorating with @tf.function
    # Removed variables include:
    # count, outputs, char_attention_plot, tag_attention_plot, dec_input
    def ____call__(self, 
                 inputs, targets, tags, 
                 char_embedder, 
                 tag_embedder, 
                 dense_fc, 
                 start_token, 
                 optimizer, 
                 mask,
                 mode, 
                 training, 
                 return_outputs, 
                 return_attention_plots,
                 count, outputs, char_attention_plot, tag_attention_plot, dec_input, loss):
        # Unpack the batched dataset
        # inputs, targets, tags = batch_dataset
        batch_size = inputs.shape[0]

        # count = tf.cast(tf.ones((batch_size,)), tf.bool) # (batch_size, 1)

        # if return_outputs and not training:
        #     outputs = tf.zeros((batch_size, 1), tf.int32) # (batch_size, 1)
        # if return_attention_plots and not training:
        #     char_attention_plot = tf.zeros((batch_size, 1, inputs.shape[1])) # (batch_size, 1, char_seq_length)
        #     tag_attention_plot = tf.zeros((batch_size, 1, tags.shape[1])) # (batch_size, 1, tag_seq_length)
        
        with tf.GradientTape() as tape:
            embedded_inputs = char_embedder(inputs, training=training)
            enc_output, enc_hidden, enc_c = self.char_encoder(embedded_inputs,
                                                    training=training)

            tag_mask = create_padding_mask(tags, 'transformer')
            embedded_tags = tag_embedder(tags, training=training)
            tag_output = self.tag_encoder(embedded_tags, training=training, mask=tag_mask)

            dec_states = (enc_hidden, enc_c)
            
            if mask == 1:
                enc_mask = create_padding_mask(inputs, 'structure')
                tag_mask = create_padding_mask(tags, 'luong')
            else:
                enc_mask, tag_mask = None, None
            
            self.decoder.reset()
            for t in range(1, targets.shape[1]):
                # embedded_inputs = char_embedder(dec_input, training=training)
                dec_output, dec_states, attention_weights = self.decoder(
                                                                embedded_inputs[:,:1,:],
                                                                dec_states,
                                                                enc_output,
                                                                tag_output,
                                                                enc_mask=enc_mask,
                                                                tag_mask=tag_mask,
                                                                training=training)
                predictions = dense_fc(dec_output, training=training)
                loss += self.loss_function(targets[:, t], predictions)
                
                if training:
                    if mode in ['P1', 'P2', 'P3']:
                        # using teacher forcing
                        dec_input = tf.expand_dims(targets[:, t], 1)
                    elif mode in ['P4']:
                        # using scheduled sampling
                        if random.random() > 0.5:
                            dec_input = tf.expand_dims(targets[:, t], 1)
                        else:
                            dec_input = tf.argmax(predictions, axis=-1, output_type=tf.int32)
                            dec_input = tf.expand_dims(dec_input, 1)
                else:
                    # calculating accuracy when running idation
                    dec_input = tf.argmax(predictions, axis=-1, output_type=tf.int32) # (batch_size,)
                    
                    mask = tf.math.equal(targets[:, t], 0)
                    accuracy = (targets[:, t] == dec_input)
                    accuracy = tf.math.logical_or(accuracy, mask)
                    count = tf.math.logical_and(count, accuracy)
                    
                    dec_input = tf.expand_dims(dec_input, 1) # (batch_size, 1)
                    if return_outputs:
                        # shape == (batch_size, t)
                        outputs = tf.concat([outputs, dec_input], axis=-1)
                    if return_attention_plots:
                        # shape == (batch_size, t, seq_length)
                        buff = tf.reshape(attention_weights[0], (batch_size, 1, -1))
                        char_attention_plot = tf.concat([char_attention_plot, buff], axis=1)

                        buff = tf.reshape(attention_weights[1], (batch_size, 1, -1))
                        tag_attention_plot = tf.concat([tag_attention_plot, buff], axis=1) # ()
                
        count = tf.reduce_sum(tf.cast(count, dtype='int32'))

        batch_loss = (loss / int(targets.shape[1]))

        if training:
            if mode == 'P1':
                variables = char_embedder.trainable_variables + tag_embedder.trainable_variables + \
                            self.char_encoder.trainable_variables + self.tag_encoder.trainable_variables + \
                            self.decoder.trainable_variables + dense_fc.trainable_variables
            elif mode == 'P2':
                variables = char_embedder.trainable_variables + tag_embedder.trainable_variables + \
                            dense_fc.trainable_variables
            elif mode == 'P3':
                variables = self.char_encoder.trainable_variables + self.tag_encoder.trainable_variables + \
                            self.decoder.trainable_variables + dense_fc.trainable_variables
            elif mode == 'P4':
                variables = dense_fc.trainable_variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

        if return_attention_plots and return_outputs:
            return batch_loss, count, outputs, (char_attention_plot, tag_attention_plot)
        elif return_attention_plots:
            return batch_loss, count, (char_attention_plot, tag_attention_plot)
        elif return_outputs:
            return batch_loss, count, outputs
        else:
            return batch_loss, count

class Run():
    def __init__(self, train_step):
        # Only train_step is needed
        # The rest are for storing history (which is not used)
        self.total_loss = None
        self.val_total_loss = None
        self.val_total_accuracy = None
        self.train_step = train_step

    def __call__(self, 
                 train_dataset, 
                 val_dataset, 
                 char_embedder, 
                 tag_embedder, 
                 dense_fc,
                 start_token, 
                 optimizer, 
                 mode='P1', 
                 mask=0, 
                 batch_size=10):
        # Batch the train and  datasets
        train_batch_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        val_batch_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        self.total_loss = 0
        
        for (batch, train_batch) in enumerate(train_batch_dataset):
            batch_loss, _ = self.train_step(train_batch,
                                       char_embedder,
                                       tag_embedder,
                                       dense_fc,
                                       start_token,
                                       optimizer,
                                       mask=mask,
                                       mode=mode,
                                       training=True)
            self.total_loss += batch_loss

            # if batch % 100 == 0:
            #     logger.debug('{} Epoch  Batch {} Loss {:.4f}'.format(
            #                                             mode,
            #                                             int(checkpoint.step),
            #                                             batch,
            #                                             batch_loss.numpy()))
        self.total_loss /= batch
        
        # Calculate idation accuracy
        self.val_total_loss = 0
        self.val_total_accuracy = 0

        for (batch, val_batch) in enumerate(val_batch_dataset):
            batch_loss, batch_accuracy = self.train_step(val_batch,
                                                    char_embedder,
                                                    tag_embedder,
                                                    dense_fc,
                                                    start_token,
                                                    optimizer,
                                                    mask=mask,
                                                    mode=mode,
                                                    training=False)
            self.val_total_loss += batch_loss
            self.val_total_accuracy += batch_accuracy
        
        self.val_total_loss /= batch
        self.val_total_accuracy /= batch * batch_size
        
        return self.total_loss, self.val_total_loss, self.val_total_accuracy
