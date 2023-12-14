# Prompt Injection possible solution.

The idea is to have a conditional transformation to the positional embedding,
such that the vectorised tokens can be put into a 'invalid' negative position.
The transformation needs to preserve as much of the token information as
possible, but also provide a strong signal that these tokens are different.
Since RoPE is a popular embedding scheme perhaps one such transformation would
be the complex conjugate (called Q* if applied to the query, which rings a bell)
Anyway, during training the model is conditioned such that for tokens in the
negative position it should predict the next token as <|endoftext|> or some
other single error token. At inference a system message is given with all
tokens in the negative position, followed by the user prompt with valid tokens
The model is able to recall system message tokens and use them in context for
future predictions however, if asked to recall these tokens verbatim, the model
will print one of the negative tokens and immediately predict the next token as
end/error, allowing the termination of output, and thus not revealing the
system prompt.
