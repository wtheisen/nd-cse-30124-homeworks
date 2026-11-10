"""HW09 explanation plates (backpropagation through time).  See plates_common.py for usage.

Nov 8 2026 rebuild on the SP26 classes: plates follow LinearLayer / Tanh / RecurrentBlock / RecurrentNeuralNetwork / CharRNN
(Wx, Wc, b; d_outputs, dcurrent, dz_at_timestep, dc_next; renamed from W, U, h the same day to match the AI2AI book: c for context). The FA26-draft plates for gather, clipping, the vanishing-gradient
experiment and the autograd check were dropped with those tasks.
"""
import math

from plates_common import *  # noqa: F401,F403
CODE_COLORS.update({'W': 'blue', 'U': 'blue', 'Wx': 'blue', 'Wc': 'blue', 'dWx': 'green', 'dWc': 'green', 'c': 'gold', 'b': 'gray', 'dW': 'green', 'dU': 'green', 'db': 'green', 'dX': 'green',
                    'logits': 'purple', 'Y': 'green', 'yb': 'green', 'Y_hat': 'gold', 'h': 'gold'})  # variables in code, by role (network colours)

PLATES, plate = plate_registry()


# ---------------------------------------------------------------- Task 01-1
@plate('plate_01-1_linear_3d')
def _():
    return title('A Linear Layer on Sequences') + f'''
<p>HW07's linear layer saw one row per image, <span class="code">(B, D)</span>. In the RNN, the output layer sees one row per <b>(phrase, position)</b>: <span class="code">(B, T, D)</span>.
Every one of those <span class="m">B × T</span> rows is an ordinary example, so flatten them into one long batch and use HW07's formulas:</p>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><h3>1. Flatten</h3>
 <div class="m" style="font-size:18px;line-height:1.8"><span class="o">dY</span> <span class="code">(B, T, Out)</span> → <span class="code">(B·T, Out)</span><br><span class="r">X</span> <span class="code">(B, T, In)</span> → <span class="code">(B·T, In)</span></div>
 <p style="font-size:15px;margin:6px 0 0" class="k"><span class="code">reshape(-1, size)</span>: the <span class="code">-1</span> means "whatever makes the count work"</p></div>
 <div class="card" style="margin:0;flex:1.2"><h3>2. HW07's three gradients</h3>
 <div class="m" style="font-size:18px;line-height:1.8"><span class="g">dW</span> = <span class="o">dY_flat</span><sup>T</sup> @ <span class="r">X_flat</span> &nbsp;<span class="code">(Out, In)</span><br>
 <span class="g">db</span> = <span class="o">dY</span>.sum(dim=(0, 1)) &nbsp;<span class="code">(Out,)</span><br>
 <span class="g">dX</span> = <span class="o">dY</span> @ <span class="b">W</span> &nbsp;<span class="code">(B, T, In)</span></div>
 <p style="font-size:15px;margin:6px 0 0" class="k"><span class="code">dX</span> needs no flattening: <span class="code">@</span> already works on the last dimension</p></div>
</div>
<div class="card" style="margin-bottom:12px"><h3>The embedding layer gets this for free</h3>
<span class="code">EmbeddingLayer</span> inherits <span class="code">backward</span>. Its cached input is one-hot, so <span class="m"><span class="o">dY_flat</span><sup>T</sup> @ <span class="r">X_flat</span></span> adds each position's gradient into the <b>column of the token that was there</b>:
a token used three times gets three contributions, summed. Padding sits after every real character, so its column only ever receives zero.</div>
<div class="callout lg fg" style="margin-bottom:0">Every gradient has the shape of the tensor it belongs to: check <span class="code">dW</span> against <span class="code">W</span>, <span class="code">db</span> against <span class="code">b</span>, <span class="code">dX</span> against <span class="code">X</span>.</div>
'''


# ---------------------------------------------------------------- Task 01-3
@plate('plate_01-3a_two_jobs')
def _():
    return title('Unrolling Turns a Loop into a Deep Network') + '''
<p>HW08's <span class="code">RecurrentBlock.forward</span> applies the same two equations at every position <span class="m">t</span>:</p>
<div class="eq" style="font-size:22px"><span class="m p">z<sub>t</sub></span> = <span class="r">x<sub>t</sub></span><span class="b">W<sub>x</sub></span><sup>T</sup> + <span class="y">c<sub>t−1</sub></span><span class="b">W<sub>c</sub></span><sup>T</sup> + <span class="k2">b</span>,&nbsp;&nbsp;
 <span class="y">c<sub>t</sub></span> = tanh(<span class="p">z<sub>t</sub></span>)</div>
<p>A 5-character phrase is a 5-layer network where <b>every layer uses the same weights</b>. Backpropagation through time (BPTT) is HW07's backward pass on that network, last position first. Two things are new:</p>
<div class="card"><h3>1. Each hidden state has two jobs, so its gradient has two paths</h3>
<div class="eq" style="font-size:22px;margin:10px 0 6px">∂<span class="ls">𝓛</span>/∂<span class="y">c<sub>t</sub></span> = <span class="code">d_outputs[:, t, :]</span> &nbsp;+&nbsp; ∂<span class="ls">𝓛</span>/∂<span class="p">z<sub>t+1</sub></span> <span class="b">W<sub>c</sub></span></div>
<div style="display:flex;justify-content:center;gap:120px;font-size:15px" class="k"><span>this position's output layer</span><span>the future: <span class="code">dc_next</span></span></div>
<p style="font-size:17px;margin:10px 0 0">The chain rule <b>adds</b> them: that sum is <span class="code">dcurrent</span>. <span class="code">dc_next</span> carries the second term backward; it starts at <b>zero</b> after the last position.</p></div>
<div class="callout lp fp" style="margin-bottom:0"><b>2. Shared weights accumulate.</b> <span class="m"><span class="b">W<sub>x</sub></span>, <span class="b">W<sub>c</sub></span>, <span class="k2">b</span></span> are used at every position, so each gradient is a <b>sum</b> of one contribution per position, like HW07's sums over a batch. Inside the loop: <span class="code">+=</span>, never <span class="code">=</span>.</div>
'''


@plate('plate_01-3b_one_step_backward')
def _():
    rows = [('<span class="code">dcurrent</span> = ', 'd_outputs[:, t, :] + dc_next', '', '(B, H)', 'c_t', 'fy'),
            ('<span class="code">dtanh</span> = ', 'activation.backward(c_at_timestep)', '1 − c²', '(B, H)', 'slope', 'fk'),
            ('<span class="code">dz_at_timestep</span> = ', 'dcurrent * dtanh', 'element-wise', '(B, H)', 'z_t', 'fp'),
            ('<span class="code">dWx</span> += ', 'dz.T @ x_at_timestep', '(H, B) @ (B, E)', '(H, E)', 'Wx', 'fb'),
            ('<span class="code">dWc</span> += ', 'dz.T @ c_prev_at_timestep', '(H, B) @ (B, H)', '(H, H)', 'Wc', 'fb'),
            ('<span class="code">db</span> += ', 'dz.sum(dim=0)', '', '(H,)', 'b', 'fk'),
            ('<span class="code">dX[:, t, :]</span> = ', 'dz @ Wx', '(B, H) @ (H, E)', '(B, E)', 'x_t', 'fr'),
            ('<span class="code">dc_next</span> = ', 'dz @ Wc', '(B, H) @ (H, H)', '(B, H)', 'c_prev', 'fk')]
    body = ''.join(
        f'<tr><td><span style="display:inline-block;width:8px;height:20px;vertical-align:middle;margin-right:8px;border-radius:2px" class="{f}"></span>{a}</td>'
        f'<td class="code">{b}</td><td class="code k">{c}</td><td class="code"><b>{d}</b> <span class="note k">= {e}</span></td></tr>'
        for a, b, c, d, e, f in rows)
    return title('One Step Backward, at Position t') + f'''
<p style="margin-bottom:8px">Read this step's cache with <span class="code">self.hidden_states[backwards_timestep + 1]</span> (entry 0 is the zero start), then:</p>
<table class="t" style="font-size:16px"><tr><th>Quantity</th><th>Formula</th><th>Shapes multiplied</th><th>Result</th></tr>{body}</table>
<div class="row">
 <div class="callout ly fy" style="margin:0">The tanh slope is <span class="m">1 − tanh(z)<sup>2</sup> = 1 − c<sub>t</sub><sup>2</sup></span>: computed from the <b>cached</b> hidden state. That's why HW08 cached every state.</div>
 <div class="callout lg fg" style="margin:0">Same pattern as HW07: weight gradient = (gradient arriving)<sup>T</sup> @ (layer input); gradient passed back = (gradient arriving) @ weight. <b>Every gradient has the shape of its tensor.</b></div>
</div>
'''


@plate('plate_01-3c_one_number_rnn')
def _():
    h1, h2 = math.tanh(0.5), math.tanh(-0.5 + 0.8 * math.tanh(0.5))
    dz2 = -0.4 * (1 - h2 ** 2)
    d1 = 0.2 + dz2 * 0.8
    dz1 = d1 * (1 - h1 ** 2)
    assert (round(h1, 3), round(h2, 3), round(dz2, 3), round(d1, 3), round(dz1, 3)) == (0.462, -0.13, -0.393, -0.115, -0.09)
    return title('Worked Example: A One-Number RNN') + f'''
<p style="font-size:18px">Hidden size 1: <span class="m b">W<sub>x</sub> = 0.5</span>, <span class="m b">W<sub>c</sub> = 0.8</span>, no bias, inputs <span class="m r">x<sub>1</sub> = 1</span>, <span class="m r">x<sub>2</sub> = −1</span>, <span class="m y">c<sub>0</sub> = 0</span>.
The output layer sends back <span class="code">d_outputs</span> <span class="m">= 0.2</span> at step 1 and <span class="m">−0.4</span> at step 2.</p>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><div class="lab b" style="color:#1155CC">forward</div>
  <div class="eqrow" style="display:block;font-size:20px"><span class="y">c<sub>1</sub></span> = tanh(0.5) = <span class="y">0.462</span></div>
  <div class="eqrow" style="display:block;font-size:20px"><span class="y">c<sub>2</sub></span> = tanh(−0.5 + 0.8 × 0.462)</div>
  <div class="eqrow" style="display:block;font-size:20px;padding-left:40px">= tanh(−0.130) = <span class="y">−0.130</span></div></div>
 <div class="card" style="margin:0"><div class="lab" style="color:#666">backward, step 2 (last: dc_next = 0)</div>
  <div class="eqrow" style="display:block;font-size:20px">dcurrent<sub>2</sub> = −0.4 + 0 = −0.4</div>
  <div class="eqrow" style="display:block;font-size:20px">dz<sub>2</sub> = −0.4 × (1 − 0.130<sup>2</sup>) = −0.393</div></div>
</div>
<div class="card"><div class="lab" style="color:#666">backward, step 1</div>
 <div class="eqrow" style="display:block;font-size:20px">dcurrent<sub>1</sub> = <span style="background:#EFEFEF;padding:2px 6px;border-radius:3px">0.2</span> + <span style="background:#D9EAD3;padding:2px 6px;border-radius:3px">(−0.393)(0.8)</span> = −0.115
  <span class="note k" style="margin-left:10px">output path + future path (dc_next)</span></div>
 <div class="eqrow" style="display:block;font-size:20px">dz<sub>1</sub> = −0.115 × (1 − 0.462<sup>2</sup>) = −0.090</div></div>
<div class="card"><div class="lab" style="color:#38761D">shared weights: one term per step, added</div>
 <div class="eqrow" style="display:block;font-size:20px">dW<sub>c</sub> = dz<sub>2</sub> c<sub>1</sub> + dz<sub>1</sub> c<sub>0</sub> = (−0.393)(0.462) + 0 = −0.182</div>
 <div class="eqrow" style="display:block;font-size:20px">dW<sub>x</sub> = dz<sub>2</sub> x<sub>2</sub> + dz<sub>1</sub> x<sub>1</sub> = 0.393 − 0.090 = 0.303</div></div>
<div class="callout lr fr" style="margin-bottom:0">Without the future path, <span class="m">dcurrent<sub>1</sub></span> would be <span class="m">+0.2</span> instead of <span class="m">−0.115</span>: the gradient would point the <b>wrong way</b>.</div>
'''


# ---------------------------------------------------------------- Task 02-1
@plate('plate_02-0_a_translation_task')
def _():
    def cells(chars, f):
        return ''.join(f'<div class="box {f}" style="min-width:0;width:54px;padding:6px 0;font-family:Inc;font-size:19px;font-weight:normal">{c}</div>' for c in chars)
    src = ['u', 'r', 'y', 'y', 'b', '␣', '6', '9', 'EOS']
    tgt = ['h', 'e', 'l', 'l', 'o', '␣', '6', '9', 'EOS']
    arrows = ''.join('<div style="width:54px;text-align:center;font-size:20px;color:#666">↓</div>' for _ in src)
    grid = f'''<div style="display:flex;flex-direction:column;align-items:center;gap:4px;margin:4px 0 14px">
 <div style="display:flex;gap:6px;align-items:center"><span class="k" style="width:120px;text-align:right;font-size:16px">encoded (input)</span>{cells(src, 'fr lr')}</div>
 <div style="display:flex;gap:6px;align-items:center"><span style="width:120px"></span>{arrows}</div>
 <div style="display:flex;gap:6px;align-items:center"><span class="k" style="width:120px;text-align:right;font-size:16px">plain (target)</span>{cells(tgt, 'fg lg')}</div>
</div>'''
    return title('The Task: Translation, One Character at a Time') + f'''
<p>Our network reads a whole encoded phrase and writes the decoded phrase, <b>one output character for every input character</b>. The two sequences line up position by position, so every position has its own answer:</p>
{grid}
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px">
 <div class="card" style="margin:0"><h3>This task: aligned translation</h3>
 <span style="font-size:16px">Input and target are the <b>same length</b>. The target at position <span class="m">t</span> is the plain character at position <span class="m">t</span>, and every real position is scored, <span class="code">&lt;EOS&gt;</span> included.</span></div>
 <div class="card" style="margin:0"><h3>Not next-character prediction</h3>
 <span style="font-size:16px">A language model (the kind behind ChatGPT) reads <span class="code">h e l l</span> and is scored on <span class="code">e l l o</span>: the target is the input <b>shifted by one</b>, a guess at what comes next.</span></div>
 <div class="card" style="margin:0"><h3>Not general translation</h3>
 <span style="font-size:16px"><span class="code">le chat noir</span> → <span class="code">the black cat</span>: the lengths can differ and the words reorder, so the model must read the whole sentence before writing (an encoder-decoder).</span></div>
</div>
<div class="callout lg fg" style="margin-bottom:0">Because the two sequences line up, the network can give its answer for position <span class="m">t</span> as soon as it has read position <span class="m">t</span>: <b>one forward pass, one prediction per position</b>, and the loss compares the two sequences position by position.</div>
'''


@plate('plate_02-1a_masked_loss')
def _():
    return title('One Prediction per Position') + '''
<p>In HW07 each image had one label. Now each phrase has a label <b>at every position</b>: probabilities <span class="code">Y_hat</span> and one-hot targets <span class="code">Y</span> are both <span class="code">(B, T, V)</span>.
The loss averages the negative log probability of the correct character over every <b>real</b> position:</p>
<div class="eq"><span class="ls">𝓛</span> = − <span class="frac"><span>1</span><span>N</span></span> Σ<sub>(i,t) : <span class="g">y<sub>it</sub></span> ≠ PAD</sub> log <span class="y">p<sub>it</sub></span>(<span class="g">y<sub>it</sub></span>)</div>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><h3>1. Every position</h3><span class="code">-(Y * log_probs).sum(dim=-1)</span>: the one-hot target zeroes every term but the correct character's, giving <span class="code">(B, T)</span></div>
 <div class="card" style="margin:0"><h3>2. Keep the real ones</h3><span class="code">Y.argmax(dim=-1) != padding_idx</span>: a <span class="code">(B, T)</span> mask; multiply by it</div>
 <div class="card" style="margin:0"><h3>3. Average</h3>divide the sum by <span class="m">N</span> = <span class="code">mask.sum()</span>, the number of real positions</div>
</div>
<div class="row">
 <div class="callout lr fr" style="margin:0"><b>Why not average over all B × T?</b> A batch with more padding would get a different loss than the same phrases batched with less, and the network would be rewarded for predicting <span class="code">&lt;PAD&gt;</span>.</div>
 <div class="callout lg fg" style="margin:0"><span class="code">&lt;EOS&gt;</span> <b>does</b> count: knowing where a phrase ends is part of the answer. <span class="code">clamp_min(1e-12)</span> keeps <span class="code">log</span> away from 0.</div>
</div>
'''


@plate('plate_02-1b_gradient_at_top')
def _():
    return title('The Gradient at the Top, One More Time') + '''
<p>HW07's logit gradient was <span class="m">(p − onehot(y)) / B</span>. With the loss averaged over <span class="m">N</span> real positions:</p>
<div style="display:flex;justify-content:center;align-items:center;gap:12px;font-family:Inc;font-size:26px;margin:6px 0 18px">
 <span>∂<span class="ls">𝓛</span>/∂<span class="p">logits<sub>it</sub></span> =</span>
 <div style="border-left:3px solid #000;border-radius:14px 0 0 14px;padding:2px 0 2px 14px;display:grid;grid-template-columns:auto auto;column-gap:28px;row-gap:8px;align-items:center">
  <span>(<span class="y">p<sub>it</sub></span> − onehot(<span class="g">y<sub>it</sub></span>)) / N</span><span class="note">real position</span>
  <span>0</span><span class="note">padded position</span></div></div>
<div class="card"><h3>Where "probability minus target" comes from</h3>
 One position's loss is <span class="m">−log p<sub>y</sub> = −z<sub>y</sub> + log Σ<sub>k</sub> e<sup>z<sub>k</sub></sup></span>, where <span class="m">z</span> are its logits. Differentiate with respect to <span class="m">z<sub>j</sub></span>:
 <div class="row" style="margin-top:10px">
  <div class="callout lr fr" style="margin:0">first term: <span class="m">−onehot(y)<sub>j</sub></span><br><span style="font-size:16px">(−1 if j = y, otherwise 0)</span></div>
  <div class="callout ly fy" style="margin:0">second term: <span class="m y">p<sub>j</sub></span><br><span style="font-size:16px">(e<sup>z<sub>j</sub></sup> / Σ<sub>k</sub> e<sup>z<sub>k</sub></sup>, the softmax)</span></div>
 </div></div>
<div class="row">
 <div class="callout lg fg" style="margin:0">Raise the correct logit, lower every other one in proportion to the probability it stole. A position already right with <span class="m">p ≈ 1</span> contributes almost nothing; a <b>confident mistake</b> contributes a lot.</div>
 <div class="callout lp fp" style="margin:0"><b>In <span class="code">backward</span>:</b> <span class="code">dA = (Y_hat - Y) * mask / normalizer</span>, then <span class="code">dA = layer.backward(dA)</span> for every layer in <b>reverse</b> order. It's the <b>only</b> place the targets enter the backward pass.</div>
</div>
'''


@plate('plate_02-1c_worked_example')
def _():
    return title('Worked Example: The Toy Phrase, N = 2') + f'''
<div class="vgrid" style="grid-template-columns:150px auto auto auto;column-gap:16px;row-gap:14px">
 <span></span><span class="lab">probabilities Y_hat</span><span class="lab">one-hot target Y</span><span class="lab">(Y_hat − Y) / 2</span>
 <span>position 0</span>{vec(['0.665', '0.245', '0.090'], 'fy', 'ly')}{vec([0, 1, 0], 'fk', 'lk', {1: 'fg'})}{vec(['0.333', '−0.378', '0.045'], 'fw', 'ld')}
 <span>position 1</span>{vec(['0.090', '0.665', '0.245'], 'fy', 'ly')}{vec([0, 0, 1], 'fk', 'lk', {2: 'fg'})}{vec(['0.045', '0.333', '−0.378'], 'fw', 'ld')}
 <span class="k">position 2 (PAD)</span>{vec(['0.007', '0.007', '0.987'], 'fk', 'lk')}<span class="note k">masked to 0</span>{vec([0, 0, 0], 'fk', 'lk')}
</div>
<div class="callout lg fg" style="margin-top:18px">The correct class gets a <b>negative</b> entry, so a downhill step raises it. Divided by <span class="m">N</span>, not <span class="m">B</span> and not <span class="m">B × T</span>.</div>
<div class="callout lr fr" style="margin-bottom:0">Padded positions get exactly <b>zero</b>: nothing the network does at a padded position can change the loss.</div>
'''


@plate('plate_02-1d_training_loop')
def _():
    steps = [('Forward', '<span class="code">forward(X, eval=True)</span><br>probabilities', 'fb lb'), ('Loss, accuracy', '<span class="code">cross_entropy</span><br><span class="code">get_accuracy</span>', 'fr lr'),
             ('Backward', '<span class="code">backward(Y_hat, Y)</span><br>stores gradients', 'fr lr'), ('Update', '<span class="code">update(lr)</span><br>for every layer', 'fp lp')]
    cyc = ''
    for i, (h, s, f) in enumerate(steps):
        if i:
            cyc += '<span class="arr">→</span>'
        cyc += f'<div class="box {f}" style="width:190px;padding:12px">{h}<small style="font-family:Arial;font-size:14px;display:block;margin-top:4px">{s}</small></div>'
    return title('The Training Loop') + f'''
<p>Every batch repeats HW07's recipe; <span class="code">train</span> runs it for every batch, every epoch:</p>
<div class="flow">{cyc}</div>
<div class="row" style="margin-top:18px">
 <div class="callout lr fr" style="margin:0"><b>Probabilities, not logits.</b> <span class="code">cross_entropy</span> takes a log of its input and <span class="code">backward</span> subtracts the one-hot target from it: both need <b>probabilities</b>, so the forward pass in <span class="code">train</span> uses <span class="code">eval=True</span>.</div>
 <div class="card" style="margin:0"><h3>Why learning rate 2.0?</h3>Much larger than HW07's: the loss is averaged over thousands of characters per batch, so each gradient is small and steady. At 0.5 it still learns, but the rare characters (digits, the end of a phrase after a number) take far more epochs.</div>
</div>
<div class="callout lp fp" style="margin-bottom:0">Each layer updates its <b>own</b> parameters from the gradients its <span class="code">backward</span> stored. <span class="code">Tanh</span> has none, so its <span class="code">update</span> does nothing; the <span class="code">RecurrentBlock</span> lives in <span class="code">self.layers</span> and updates <span class="code">Wx</span>, <span class="code">Wc</span> and <span class="code">b</span>.</div>
'''


# ---------------------------------------------------------------- Task 03
@plate('plate_03-1_what_to_expect')
def _():
    stages = [('epoch 1', 'loss starts near log 41 ≈ 3.7 (uniform guessing); accuracy is near 90% by the end of the epoch', 'spaces stay spaces; most of the letter substitution', 'fk lk'),
              ('by epoch 13', 'training accuracy passes 99%', 'the rest of the letters; the note reads correctly', 'fb lb'),
              ('after that', 'loss keeps falling, accuracy creeps toward 100%', 'the rare cases: digits, and where a phrase ends after a number', 'fg lg')]
    rows = ''.join(f'<div class="box {f}" style="font-weight:normal;text-align:left;padding:10px 14px;font-size:16px"><span class="n" style="font-size:15px">{a.upper()}</span><br><b>{b}</b><br><span class="k">learns: {c}</span></div>' for a, b, c, f in stages)
    return title('What to Expect') + f'''
<div class="row" style="margin-bottom:14px">
 <div class="card" style="margin:0"><h3>Updates</h3><div class="m">2,000 phrases / 64 → 32 batches</div><div class="m">32 × 50 epochs = <b>1,600 updates</b></div></div>
 <div class="card" style="margin:0"><h3>Progress reports</h3><span class="code">train</span> prints five times (epochs 1, 13, 25, 37 and 49), each with the network's current reading of the note. Exact numbers depend on the random starting weights.</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">{rows}</div>
<div class="callout ly fy" style="margin-bottom:0;margin-top:14px"><b>Training accuracy isn't the whole story.</b> The supplied code also reports accuracy on the 200 validation phrases, which training never sees.</div>
'''


@plate('plate_03-2_library_loop')
def _():
    rows = [('Clear old gradients', 'rnn_optimizer.zero_grad()'),
            ('Forward', 'logits = rnn_model(xb)'),
            ('Flatten', 'logits.reshape(B * T, V), yb.reshape(B * T)'),
            ('Loss', 'loss = rnn_loss_fn(logits_flat, yb_flat)'),
            ('Backward', 'loss.backward()'),
            ('Update', 'rnn_optimizer.step()')]
    body = ''.join(f'<tr><td><b>{a}</b></td><td class="code" style="font-size:15px">{b}</td></tr>' for a, b in rows)
    return title('The Same Loop, the Library Way') + f'''
<p>HW07's PyTorch pattern, with two sequence-specific changes:</p>
<table class="t"><tr><th>Step</th><th>PyTorch</th></tr>{body}</table>
<div class="row">
 <div class="callout lb fb" style="margin:0"><b>The loss.</b> <span class="code">nn.CrossEntropyLoss</span> takes logits (it applies the softmax itself) and class <b>IDs</b>, one row per prediction: hence the flattening. <span class="code">ignore_index</span> set to the padding ID does exactly what your mask does.</div>
 <div class="callout ly fy" style="margin:0"><b>Adam, not plain gradient descent.</b> <span class="code">torch.optim.Adam</span> adapts the step size for every parameter, so learning rate 0.001 is enough. Your scratch network uses plain steps at 2.0.</div>
</div>
<div class="callout lg fg" style="margin-bottom:0">Different optimizers and different starting weights, the same architecture: both should end up reading the note correctly.</div>
'''


# ---------------------------------------------------------------- Task 04-1
@plate('plate_04-1a_two_accuracies')
def _():
    return title('Opening the Test Set, Once') + f'''
<p>Neither network has seen the 200 <b>test</b> phrases. Evaluate both on them, once, and then on the note.</p>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><h3>Token accuracy</h3>fraction of real characters (including <span class="code">&lt;EOS&gt;</span>) predicted correctly: what <span class="code">get_accuracy</span> computes</div>
 <div class="card" style="margin:0"><h3>Exact-phrase accuracy</h3>fraction of phrases with <b>every</b> character correct</div>
</div>
<div class="card"><h3>They can tell very different stories</h3>
<div class="eqrow" style="display:block;font-size:18px">30 characters, one wrong: <span class="g">29/30 ≈ 97%</span> by tokens, <span class="r">a complete miss</span> by phrases</div>
<p style="font-size:17px;margin:8px 0 0">For evidence the second number often matters more: one wrong digit in a locker number is one wrong locker.</p></div>
<div class="callout lr fr" style="margin-bottom:0"><b>One look only.</b> Change the learning rate because the test number disappointed you, and the test set quietly becomes a validation set: its numbers would overstate how well the decoder handles evidence it has never seen.</div>
'''


@plate('plate_04-1b_how_the_network_reads')
def _():
    return title('How the Network Reads the Note') + f'''
{chain([('encode the note', 'fb lb'), ('one forward pass', 'fp lp'), ('argmax at every position', 'fy ly'), ('stop at first &lt;EOS&gt;', 'fg lg')], size=17)}
<div class="row" style="margin-top:18px;margin-bottom:12px">
 <div class="card" style="margin:0"><h3>Greedy decoding</h3>The single most likely plain character, <b>independently</b> at each step. Nothing forces valid English: a badly learned substitution would show up immediately as garbled letters. HW10's attention model reads the same way, so the two are easy to compare.</div>
 <div class="card" style="margin:0"><h3>Checking against the answer key</h3>ROT13 is exact: shift each letter 13 places, leave spaces and digits alone; applying it again undoes it. <span class="code">codecs.encode(text, 'rot_13')</span> lets us check the network's reading <b>character by character</b>.</div>
</div>
<div class="callout lp fp" style="margin-bottom:0">Why train a network at all? The decoder we can verify here is the one we'd have to trust on evidence we <b>can't</b> check by hand.</div>
'''


SPEC = {'### Task 01-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_01-1_linear_3d',
                                            "A linear layer on sequences: flatten dY and X across batch and position, then apply HW07's "
                                            'dW, db, dX formulas; the embedding layer inherits the same backward.')])],
 '### Task 01-3: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_01-3a_two_jobs',
                                            "Unrolling the recurrent block: each hidden state feeds this position's output and the next "
                                            'step, so its gradient is the sum of both paths, and the shared Wx, Wc, b gradients accumulate '
                                            'over positions.'),
                                           ('plate_01-3b_one_step_backward',
                                            'Table of one backward step at position t: dcurrent, the tanh slope, dz, the dWx, dWc, db '
                                            'contributions, dX, and dc_next, with the shapes multiplied.'),
                                           ('plate_01-3c_one_number_rnn',
                                            'Worked example of BPTT on a hidden-size-1 RNN over two steps, showing the future path '
                                            "changing the sign of the first step's gradient.")])],
 '### Task 02-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_02-1a_masked_loss',
                                            'Cross-entropy averaged over real positions only: per-position loss from one-hot targets, a '
                                            'mask from the padding ID, divided by the number of real positions.'),
                                           ('plate_02-1b_gradient_at_top',
                                            'The logit gradient is (probabilities minus one-hot target) divided by N at real positions and '
                                            'zero at padding, derived from the softmax.'),
                                           ('plate_02-1c_worked_example',
                                            'Worked example on a three-position toy phrase: probabilities, one-hot targets, and the '
                                            'resulting gradient with the padded position zeroed.'),
                                           ('plate_02-1d_training_loop',
                                            'The training loop: forward as probabilities, loss and accuracy, backward, then update every '
                                            'layer, with learning rate 2.0.')])],
 '### Task 03-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_03-1_what_to_expect',
                                            'What training looks like: 32 batches per epoch for 1,600 updates; the loss starts near 3.7 '
                                            'and training accuracy passes 99% by epoch 13; digits and phrase endings come last.')])],
 '### Task 03-2: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_03-2_library_loop',
                                            'The same training loop in PyTorch: zero_grad, forward, flatten, CrossEntropyLoss with '
                                            'ignore_index, backward, Adam step.')])],
 '### Task 04-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_04-1a_two_accuracies',
                                            'Token accuracy versus exact-phrase accuracy: one wrong character in 30 is 97% by tokens but a '
                                            'complete miss by phrases. Evaluate on the test set once.'),
                                           ('plate_04-1b_how_the_network_reads',
                                            'Greedy decoding: encode the note, one forward pass, argmax at every position, stop at the '
                                            'first EOS; checked against codecs ROT13.')])]}

if __name__ == '__main__':
    main('09', PLATES, SPEC)
