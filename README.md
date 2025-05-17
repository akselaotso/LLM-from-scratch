# LLM from scratch

Implementing an LLM "from scratch". 

Uses PyTorch modules like nn.Linear or nn.Embedding as well as ready made AdamW optimizer and cross entropy loss. Have already implemented these from scratch, without PyTorch, in my "neural network from scratch" repository. Dataset used is "The Complete Works of William Shakespeare", which is too small for any useful learning.

Includes implementations for MultiHeadAttention, text generation (with temperature scaling, top-k and multinomial sampling) and more. Opted for very simple training loop since measurable goals are not the focus.

Transformer and LLM implemented with GPT architecture.

The LLM is not very good, mostly spouting incoherent sentences. However, the goal of this project was to learn about LLM's and this goal has been achieved; I successfully built and trained an LLM to spout out random, incoherent shakespearean text. 

Other repositories might include projects with more concrete goals.

