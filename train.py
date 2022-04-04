import torch
from d2l import torch as d2l
from model import TransformerEncoder, TransformerDecoder, EncoderDecoder, MaskedSoftmaxCELoss
from utils import xavier_init_weights, grad_clipping

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu(2)
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(len(src_vocab), query_size, key_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), query_size, key_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)

net = EncoderDecoder(encoder, decoder)
net.apply(xavier_init_weights)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = MaskedSoftmaxCELoss()
net.train()

for epoch in range(num_epochs):
    metrics = {'loss': 0., 'tokens': 0.}
    for batch in train_iter:
        optimizer.zero_grad()
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
        bos = torch.tensor([tgt_vocab['<bos>']]* Y.shape[0], device=device).reshape(-1, 1)
        dec_input = torch.cat([bos, Y[:, :-1]], dim=1)
        Y_hat, _ = net(X, dec_input, X_valid_len)
        l = loss(Y_hat, Y, Y_valid_len)
        l.sum().backward()
        grad_clipping(net, 1)
        num_tokens = Y_valid_len.sum()
        optimizer.step()
        with torch.no_grad():  
            metrics['loss'] += l.sum()
            metrics['tokens'] += num_tokens
    if (epoch + 1) % 10 == 0:
        print(f"[LOG] epoch: {epoch + 1}  avg_loss: {metrics['loss'] / metrics['tokens']}")




