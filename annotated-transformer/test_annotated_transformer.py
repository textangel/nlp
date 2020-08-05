import unittest
from annotated_transformer import *

class TestAnnotatedTransformer(unittest.TestCase):

    # def test_clones(self):
    #     assert False


    def test_subsequent_mask(self):
        mask = subsequent_mask(5)
        gold = np.array(
            [[1,0,0,0,0],
             [1,1,0,0,0],
             [1,1,1,0,0],
             [1,1,1,1,0],
             [1,1,1,1,1]]
        )
        print(mask.numpy())
        assert (mask.squeeze(-1).numpy() == gold).all()

    def test_positional_encoding(self):
        pe = PositionalEncoding(20, 0, max_len=30)
        assert
    #
    # def test_attention(self):
    #     assert False
    #
    #
    # def test_make_model(self):
    #     assert False
    #
    #
    # def test_run_epoch(self):
    #     assert False
    #
    #
    # def test_batch_size_fn(self):
    #     assert False
    #
    #
    # def test_get_std_opt(self):
    #     assert False
    #
    #
    # def test_data_gen(self):
    #     assert False
    #
    #
    # def test_train_simple_copy(self):
    #     assert False
    #
    #
    # def test_greedy_decode(self):
    #     assert False
    #
    #
    # def test__run_greedy_decode(self):
    #     assert False
    #
    #
    # def test__load_data(self):
    #     assert False
    #
    #
    # def test_rebatch(self):
    #     assert False
    #
    #
    # def test_train_multigpu(self):
    #     assert False
    #
    #
    # def test_encoder_decoder(self):
    #     assert False
    #
    #
    # def test_generator(self):
    #     assert False
    #
    #
    # def test_encoder(self):
    #     assert False
    #
    #
    # def test_layer_norm(self):
    #     assert False
    #
    #
    # def test_sublayer_connection(self):
    #     assert False
    #
    #
    # def test_encoder_layer(self):
    #     assert False
    #
    #
    # def test_decoder(self):
    #     assert False
    #
    #
    # def test_decoder_layer(self):
    #     assert False
    #
    #
    # def test_multi_headed_attention(self):
    #     assert False
    #
    #
    # def test_positionwise_feed_forward(self):
    #     assert False
    #
    #
    # def test_embeddings(self):
    #     assert False
    #
    #
    # def test_batch(self):
    #     assert False
    #
    #
    # def test_noam_opt(self):
    #     assert False
    #
    #
    # def test_label_smoothing(self):
    #     assert False
    #
    #
    # def test_simple_loss_compute(self):
    #     assert False
    #
    #
    # def test_my_iterator(self):
    #     assert False
    #
    #
    # def test_multi_gpuloss_compute(self):
    #     assert False

if __name__ == '__main__':
    unittest.main()