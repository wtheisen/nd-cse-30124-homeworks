"""HW10 explanation plates (self-attention and the Transformer encoder).

    python plates10.py            # render every plate into ../imgs/homework10/plates/
    python plates10.py 01-1       # only plates whose name contains "01-1"
    python plates10.py --apply    # swap the plates into homework10.ipynb and homework10_solutions.ipynb
"""
import math

from plates_common import *
CODE_COLORS.update({'X': 'red', 'tokens': 'red', 'Wq': 'blue', 'Wk': 'blue', 'Wv': 'blue', 'Wo': 'blue', 'W1': 'blue', 'W2': 'blue', 'Wout': 'blue', 'W': 'blue', 'b1': 'gray', 'b2': 'gray', 'bout': 'gray', 'A': 'gold'})  # variables in code, by role (network colours)

PLATES, plate = plate_registry()


# Colours (Eric's network convention): inputs X red, weights Ω blue, biases grey, raw scores S purple,
# attention weights A gold (softmax outputs); gradients in the book's style (red ℒ, symbols coloured, values black).
# Q, K, V get their own colours so they stay apart: Q navy, K teal, V green.
QC, QF, QL = '#002B5B', '#D6DEE8', '#7F95AD'
KC, KF, KL = '#3F96A8', '#D7EEF2', '#8CC6D2'


def q(s):
    return f'<span style="color:{QC};font-weight:bold">{s}</span>'


def k(s):
    return f'<span style="color:{KC};font-weight:bold">{s}</span>'


def w(s):
    return f'<span class="b">{s}</span>'


def v(s):
    return f'<span class="g">{s}</span>'


def a(s):
    return f'<span class="y">{s}</span>'


# ---------------------------------------------------------------- Task 01-1
@plate('plate_01-1a_attention_context')
def _():
    def row(y, label, arcs, chain_arrows):
        out = f'<text x="10" y="{y+6}" font-family="Arial" font-size="15" font-weight="bold" fill="#002B5B">{label}</text>'
        for i in range(8):
            x = 150 + i * 92
            hot = i in (2, 7)
            mid = chain_arrows and 2 < i < 7
            fill = '#F4CCCC' if hot else ('#FFF2CC' if mid else '#EFEFEF')
            line = '#E06666' if hot else ('#FFD966' if mid else '#999')
            out += f'<rect x="{x-26}" y="{y-20}" width="52" height="40" rx="6" fill="{fill}" stroke="{line}" stroke-width="2"/>'
            out += f'<text x="{x}" y="{y+6}" text-anchor="middle" font-family="Inc" font-size="18">{i}</text>'
            if chain_arrows and i < 7:
                col = '#DCB439' if 2 <= i < 7 else '#999'
                out += f'<line x1="{x+28}" y1="{y}" x2="{x+60}" y2="{y}" stroke="{col}" stroke-width="2.5" marker-end="url(#ar{"r" if col == "#DCB439" else "k"})"/>'
        out += arcs
        return out
    x2, x7 = 150 + 2 * 92, 150 + 7 * 92
    faint = ''.join(f'<path d="M {150+j*92} 150 Q {(150+j*92+x7)/2} {150-18-abs(7-j)*9} {x7} 150" fill="none" stroke="#FFD966" stroke-width="1.5"/>'
                    for j in range(8) if j not in (2, 7))
    arcs = faint + f'<path d="M {x2} 150 Q {(x2+x7)/2} 60 {x7-4} 146" fill="none" stroke="#DCB439" stroke-width="3.5" marker-end="url(#ary)"/>'
    svg = f'''<svg width="920" height="200" viewBox="0 0 920 200">
<defs>
 <marker id="arr" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#DCB439"/></marker>
 <marker id="ark" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#999"/></marker>
 <marker id="ary" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 z" fill="#DCB439"/></marker>
</defs>
{row(30, 'HW08 RNN', '', True)}
{row(170, 'Attention', arcs, False)}
</svg>'''
    return title('A Different Way to Use Context') + f'''
<p>For position 7 to use what's at position 2, the <b>RNN</b> had to carry it through the hidden state across positions 3 to 6, in order.
<b class="def">Self-attention</b> lets every position look at <b>every</b> position at once (itself included) and decide <b>how much</b> of each to use.</p>
<div style="text-align:center;margin:4px 0 6px">{svg}</div>
<p style="margin-bottom:8px">Each position's vector <span class="m r">x<sub>i</sub></span> is projected three ways, with three learned weight matrices:</p>
<table class="t">
<tr><th>Projection</th><th>Formula</th><th>Role</th><th>Library catalog</th></tr>
<tr><td><b class="def">Query</b> <span class="m">{q('q<sub>i</sub>')}</span></td><td class="m"><span class="r">x<sub>i</sub></span>{w('W<sub>Q</sub>')}<sup>T</sup></td><td>"What am I looking for?"</td><td>your search</td></tr>
<tr><td><b class="def">Key</b> <span class="m">{k('k<sub>j</sub>')}</span></td><td class="m"><span class="r">x<sub>j</sub></span>{w('W<sub>K</sub>')}<sup>T</sup></td><td>"What do I contain, for matching?"</td><td>the catalog entries</td></tr>
<tr><td><b class="def">Value</b> <span class="m">{v('v<sub>j</sub>')}</span></td><td class="m"><span class="r">x<sub>j</sub></span>{w('W<sub>V</sub>')}<sup>T</sup></td><td>"What do I hand over if attended to?"</td><td>the books themselves</td></tr>
</table>
<div class="callout ly fy" style="margin-bottom:0">Compare your search against every entry, and take more from the books whose entries match best.</div>
'''


@plate('plate_01-1b_scores_softmax_mask')
def _():
    T, pad = 5, (3, 4)
    cells = ''
    for i in range(T):
        for j in range(T):
            if j in pad:
                cls, txt = 'background:#CCCCCC;color:#666', '0'
            else:
                cls, txt = 'background:#FFF2CC', 'a'
            if i in pad:
                cls += ';opacity:.45'
            cells += f'<div style="{cls};border:1px solid #fff;height:34px;display:flex;align-items:center;justify-content:center;font-family:Inc;font-size:17px"><span>{txt}<sub>{i}{j}</sub></span></div>' if txt == 'a' else \
                     f'<div style="{cls};border:1px solid #fff;height:34px;display:flex;align-items:center;justify-content:center;font-family:Inc;font-size:17px">{txt}</div>'
    labels_top = ''.join(f'<div style="text-align:center;font-family:Inc;font-size:14px;color:{"#666" if j in pad else KC}">k{j}{"·PAD" if j in pad else ""}</div>' for j in range(T))
    labels_side = ''.join(f'<div style="height:34px;display:flex;align-items:center;justify-content:flex-end;font-family:Inc;font-size:14px;color:{"#666" if i in pad else QC};padding-right:6px">q{i}</div>' for i in range(T))
    grid = f'''<div style="display:grid;grid-template-columns:44px 300px;gap:0">
      <div></div><div style="display:grid;grid-template-columns:repeat({T},1fr)">{labels_top}</div>
      <div>{labels_side}</div><div style="display:grid;grid-template-columns:repeat({T},1fr)">{cells}</div></div>'''
    return title('Scores, Scaling, Softmax, and Masking') + f'''
<p>Query <span class="m">i</span> compares itself with every key using a dot product, scaled by the vector width <span class="m">d</span>:</p>
<div class="eq"><span class="p">s<sub>ij</sub></span> = <span class="frac"><span>{q('q<sub>i</sub>')} · {k('k<sub>j</sub>')}</span><span>√d</span></span>
 &nbsp;&nbsp;→&nbsp;&nbsp; <span class="y">a<sub>ij</sub></span> = softmax over keys <span class="m">j</span></div>
<div class="row" style="margin-bottom:12px">
 <div class="callout lr fr" style="margin:0"><b>Why divide by √d?</b> A dot product adds up <span class="m">d</span> terms, so it grows with width. Big scores push softmax to nearly one-hot, where almost every gradient is near zero and training stalls.</div>
 <div class="callout ly fy" style="margin:0"><b>Softmax over the keys</b> (HW06) turns each row of scores into attention weights: non-negative, and each <b>row sums to 1</b>.</div>
</div>
<div class="row" style="align-items:center">
 <div style="flex:0 0 360px">{grid}<div class="lab" style="text-align:center;margin-top:6px">attention weights A, (B, T, T)</div></div>
 <div style="font-size:17px;line-height:1.5">
  <p style="font-size:17px"><b>Masking padding.</b> A real character must not take information from <span class="code">&lt;PAD&gt;</span>.
  Before softmax, set padded <b>keys'</b> scores (the columns) to <span class="m">−∞</span>: <span class="m">e<sup>−∞</sup> = 0</span>, so they get exactly zero weight and the rest still sum to 1.</p>
  <p style="font-size:17px;color:#444">Padded <b>query</b> rows (faded) still produce output, but the loss ignores them.</p>
  <p style="font-size:17px;margin:0;color:#444">No causal mask: every cipher character is available at once, so looking ahead leaks nothing. A text <i>generator</i> like GPT does add one.</p>
 </div>
</div>
'''


@plate('plate_01-1c_worked_example')
def _():
    rows = [(f"{q('q')} · {k('k<sub>j</sub>')}", ['1', '0', '1'], ''),
            ('÷ √2', ['0.7071', '0', '0.7071'], ''),
            ('softmax', ['0.4011', '0.1978', '0.4011'], 'y'),
            ('softmax, key 2 masked', ['0.6698', '0.3302', '0'], 'y')]
    body = ''
    for name, vals, cls in rows:
        tds = ''
        for j, val in enumerate(vals):
            masked = name.startswith('softmax, key') and j == 2
            style = 'background:#EFEFEF;color:#666' if masked else ('background:#FFF9E6' if cls else 'color:#674EA7;font-weight:bold')
            tds += f'<td class="m num" style="{style}{";font-weight:bold" if cls and not masked else ""}">{val}</td>'
        body += f'<tr><td class="m" style="width:300px">{name}</td>{tds}</tr>'
    return title('Worked Example: One Query, Three Keys') + f'''
<p>Width <span class="m">d = 2</span>, three positions. Position 0's query, and the three keys:</p>
<div style="display:flex;justify-content:space-around;margin:4px 0 16px" class="eqrow">
 <span>{q('q')} = {q('[1, 0]')}</span><span>{k('k<sub>0</sub>')} = {k('[1, 0]')}</span><span>{k('k<sub>1</sub>')} = {k('[0, 1]')}</span><span>{k('k<sub>2</sub>')} = {k('[1, 1]')}</span>
</div>
<table class="t"><tr><th></th><th class="num">key 0</th><th class="num">key 1</th><th class="num">key 2</th></tr>{body}</table>
<div class="row">
 <div class="callout lg fg" style="margin:0">Keys 0 and 2 match the query equally well, so they get <b>equal weight</b>. Each row sums to 1.</div>
 <div class="callout lk fk" style="margin:0">Masking key 2 (padding) sends its share to the others: <span class="m">0.6698 + 0.3302 = 1</span>.</div>
</div>
'''


# ---------------------------------------------------------------- Task 01-2
@plate('plate_01-2a_attention_forward')
def _():
    flow = chain([('<span class="r">X</span><small>(B, T, D)</small>', 'fr lr'),
                  (f'{q("Q")}, {k("K")}, {v("V")}<small>(B, T, D)</small>', 'fk lk'),
                  ('<span class="y">A</span><small>(B, T, T)</small>', 'fy ly'),
                  ('C = <span class="y">A</span><span class="g">V</span><small>(B, T, D)</small>', 'fg lg'),
                  ('Y = C<span class="b">W<sub>O</sub></span><sup>T</sup><small>(B, T, D)</small>', 'fk lk')], size=17)
    return title('From Weights to New Representations') + f'''
<p>The attention weights say <b>how much</b> to take from each position. Position <span class="m">i</span>'s output is the weighted average of the <b>values</b>, then an output projection mixes it back into the model's width:</p>
<div class="eq" style="font-size:25px"><span>c<sub>i</sub> = Σ<sub>j</sub> {a('a<sub>ij</sub>')} {v('v<sub>j</sub>')}</span> &nbsp;&nbsp; <span class="note k" style="font-size:17px">or, for all positions,</span> &nbsp;&nbsp; <span>C = {a('A')}{v('V')}</span>, &nbsp; <span>Y = C{w('W<sub>O</sub>')}<sup>T</sup></span></div>
<div class="card" style="text-align:center">
<div class="eqrow" style="justify-content:center;font-size:20px;margin-bottom:12px;gap:14px">
 <span>{q('Q')} = <span class="r">X</span>{w('W<sub>Q</sub>')}<sup>T</sup></span><span>{k('K')} = <span class="r">X</span>{w('W<sub>K</sub>')}<sup>T</sup></span><span>{v('V')} = <span class="r">X</span>{w('W<sub>V</sub>')}<sup>T</sup></span>
 <span>{a('A')} = softmax({q('Q')}{k('K')}<sup>T</sup>/√D)</span><span>Y = ({a('A')}{v('V')}){w('W<sub>O</sub>')}<sup>T</sup></span></div>
{flow}</div>
<table class="t" style="font-size:16px"><tr><th>Tensor</th><th>Shape</th><th>Meaning</th></tr>
<tr><td class="code">X</td><td class="code">(B, T, D)</td><td>one vector per position</td></tr>
<tr><td class="code">Q, K, V</td><td class="code">(B, T, D)</td><td>three projections of every position</td></tr>
<tr><td class="code">A</td><td class="code">(B, T, T)</td><td>row <span class="m">i</span>: how query <span class="m">i</span> splits its attention over all keys</td></tr>
<tr><td class="code">C = A @ V</td><td class="code">(B, T, D)</td><td>each position's weighted mix of values</td></tr>
<tr><td class="code">Y</td><td class="code">(B, T, D)</td><td>same shape as the input, so blocks can be <b>stacked</b></td></tr></table>
'''


@plate('plate_01-2b_worked_example')
def _():
    return title('Worked Example: Mixing the Values') + f'''
<p>Continuing Task 01-1: query 0's weights <span class="m y">[0.4011, 0.1978, 0.4011]</span> mix the three values
<span class="m">{v('v<sub>0</sub> = [1, 0]')}</span>, <span class="m">{v('v<sub>1</sub> = [0, 2]')}</span>, <span class="m">{v('v<sub>2</sub> = [3, 1]')}</span>:</p>
<div class="card"><div class="eqrow" style="display:block;text-align:center;font-size:22px;line-height:1.9">
 c<sub>0</sub> = {a('0.4011')}·{v('[1, 0]')} + {a('0.1978')}·{v('[0, 2]')} + {a('0.4011')}·{v('[3, 1]')}<br>
 = [0.4011, 0] + [0, 0.3956] + [1.2033, 0.4011] = <b>[1.6044, 0.7967]</b></div></div>
<div class="row" style="margin-top:4px">
 <div class="callout lp fp" style="margin:0"><b>Our teaching version</b> has <b>one head</b> (one set of <span class="m b">W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></span>) and no biases.
 We check it against <span class="code">F.scaled_dot_product_attention</span>.</div>
 <div class="callout lr fr" style="margin:0"><b>Mask convention flips.</b> PyTorch's mask is <span class="code">True</span> where a position <b>may</b> be attended to, the opposite of our <span class="code">pad_mask</span>.</div>
</div>
<div class="callout lb fb" style="margin-bottom:0"><b>Caching for the backward pass:</b> like HW07's layers, <span class="code">attention_forward</span> returns everything backward will need: the input, <span class="m">Q, K, V</span>, the weights, the context, and the four weight matrices.</div>
'''


# ---------------------------------------------------------------- Task 02-1
@plate('plate_02-1a_attention_backward')
def _():
    rows = [('Y = C <span class="b">W<sub>O</sub></span><sup>T</sup>', 'd<span class="b">W<sub>O</sub></span> = (∇<sub>Y</sub><span class="ls">𝓛</span>)<sup>T</sup>C,&nbsp; dC = (∇<sub>Y</sub><span class="ls">𝓛</span>)<span class="b">W<sub>O</sub></span>', "HW07's linear layer"),
            (f'C = {a("A")} {v("V")}', f'd<span class="y">A</span> = dC {v("V")}<sup>T</sup>,&nbsp; d{v("V")} = <span class="y">A</span><sup>T</sup>dC', 'matrix-product rule, both directions'),
            ('<span class="y">A</span> = softmax(<span class="p">S</span>)', 'd<span class="p">S</span> = <span class="y">A</span> ⊙ (d<span class="y">A</span> − Σ<sub>j</sub> d<span class="y">A<sub>ij</sub></span><span class="y">A<sub>ij</sub></span>)', 'the softmax rule'),
            (f'<span class="p">S</span> = {q("Q")}{k("K")}<sup>T</sup>/√d', f'd{q("Q")} = d<span class="p">S</span> {k("K")}/√d,&nbsp; d{k("K")} = d<span class="p">S</span><sup>T</sup>{q("Q")}/√d', 'matrix-product rule'),
            (f'{q("Q")} = <span class="r">X</span><span class="b">W<sub>Q</sub></span><sup>T</sup> (and {k("K")}, {v("V")})', f'd<span class="b">W<sub>Q</sub></span> = d{q("Q")}<sup>T</sup><span class="r">X</span>,&nbsp; d<span class="r">X</span> = d{q("Q")} <span class="b">W<sub>Q</sub></span> + d{k("K")} <span class="b">W<sub>K</sub></span> + d{v("V")} <span class="b">W<sub>V</sub></span>', "HW07's linear layer, three times")]
    body = ''.join(f'<tr><td class="m">{f}</td><td class="m">{b}</td><td style="font-size:15px">{w}</td></tr>' for f, b, w in rows)
    return title('Following the Gradient Backward') + f'''
<p>We receive the upstream gradient <span class="m">∇<sub>Y</sub><span class="ls">𝓛</span> = ∂<span class="ls">𝓛</span>/∂Y</span> and walk back through every forward step, top to bottom, using the cache:</p>
<table class="t" style="font-size:16px"><tr><th>Forward step</th><th>Backward step</th><th>Where it comes from</th></tr>{body}</table>
<div class="row">
 <div class="callout lb fb" style="margin:0"><b>The input feeds three paths.</b> <span class="m r">X</span> made <span class="m">{q("Q")}</span>, <span class="m">{k("K")}</span> <b>and</b> <span class="m">{v("V")}</span>, so <span class="m">d<span class="r">X</span></span> is the <b>sum</b> of all three, just as HW09's shared RNN weights summed over time.</div>
 <div class="callout lp fp" style="margin:0"><b>Weight gradients sum over every example and every position:</b> reshape to <span class="code">(B·T, D)</span> and one matrix product does it.</div>
</div>
'''


@plate('plate_02-1b_softmax_rule')
def _():
    return title('The Softmax Rule, One Row at a Time') + f'''
<p>Each weight depends on <b>every</b> score in its row (they share a denominator), so softmax's full derivative per row is a <span class="m">T × T</span> matrix.
Multiplying through by the upstream gradient collapses it to:</p>
<div class="eq">d<span class="p">S<sub>ij</sub></span> = <span class="y">A<sub>ij</sub></span>( d<span class="y">A<sub>ij</sub></span> − Σ<sub>k</sub> d<span class="y">A<sub>ik</sub></span> <span class="y">A<sub>ik</sub></span> )</div>
<p style="font-size:18px">Each score's gradient is its weight times "how much its upstream gradient beats the row's <b>weighted average</b>." We never build the <span class="m">T × T</span> matrix:
<span class="code">dA * A</span> is element-wise, and <span class="code">.sum(dim=-1, keepdim=True)</span> gives each row's average as a column that broadcasts back.</p>
<div class="card"><h3>Worked Example: weights a = [0.7, 0.2, 0.1], upstream dA = [1, 0, 0]</h3>
<div class="vgrid" style="grid-template-columns:230px auto 1fr">
 <span>weights <span class="m y">a</span></span>{vec(['0.7', '0.2', '0.1'], 'fy', 'ly')}<span></span>
 <span>upstream <span class="m">d<span class="y">A</span></span></span>{vec([1, 0, 0], 'fw', 'ld')}<span class="note">weighted average: <span class="m">1 × 0.7 = 0.7</span></span>
 <span>score gradient <span class="m">d<span class="p">S</span></span></span>{vec(['0.21', '−0.14', '−0.07'], 'fw', 'ld')}<span class="note m" style="font-size:16px">0.7(1−0.7), 0.2(0−0.7), 0.1(0−0.7)</span>
</div></div>
<div class="row">
 <div class="callout lg fg" style="margin:0"><b>Each row's gradients sum to zero:</b> adding a constant to every score in a row doesn't change softmax, so scores can only trade weight among themselves.</div>
 <div class="callout lk fk" style="margin:0"><b>Masked keys get nothing.</b> Their <span class="m">A<sub>ij</sub> = 0</span>, so their score gradient is exactly zero: padding gets no gradient at all.</div>
</div>
'''


# ---------------------------------------------------------------- Task 03-1
@plate('plate_03-1a_positional_encoding')
def _():
    word = lambda ws, cls: ''.join(f'<div class="box {cls}" style="min-width:0;padding:6px 12px;font-size:17px">{w}</div>' for w in ws)
    return title('Attention Has No Sense of Order') + f'''
<p>Nothing in the attention formula says <b>where</b> a position is. Shuffle the characters and every query finds the same keys with the same scores; the outputs just come out shuffled the same way.
Attention treats its input as a <b>set</b>, not a sequence.</p>
<div class="row" style="align-items:center;margin-bottom:12px">
 <div style="text-align:center"><div style="display:flex;gap:8px;justify-content:center">{word(['dog', 'bites', 'man'], 'fr lr')}</div></div>
 <div style="flex:0 0 140px;text-align:center;font-size:16px"><b>look the same<br>to attention</b><div class="arr">⇄</div></div>
 <div style="text-align:center"><div style="display:flex;gap:8px;justify-content:center">{word(['man', 'bites', 'dog'], 'fr lr')}</div></div>
</div>
<p>The RNN never had this problem: order was built into its left-to-right loop. A Transformer must be <b>told</b> where each position is, by <b>adding</b> a fixed <b class="def">positional encoding</b> to each token's embedding before the first block.</p>
<div class="card"><h3>Sinusoidal positional encodings</h3>
<div class="eq" style="font-size:22px;margin:6px 0 10px"><span>PE<sub>t,2k</sub> = sin(t ω<sub>k</sub>)</span>, &nbsp; <span>PE<sub>t,2k+1</sub> = cos(t ω<sub>k</sub>)</span>, &nbsp; <span>ω<sub>k</sub> = 10000<sup>−2k/D</sup></span></div>
<p style="font-size:17px;margin:0">Position <span class="m">t</span>, channel pair <span class="m">k</span> (channels <span class="m">2k</span> and <span class="m">2k+1</span>), model width <span class="m">D</span>.
Fixed formulas, so they work for any length, including the 262-character note, longer than any training phrase.</p></div>
<div class="callout lb fb" style="margin-bottom:0">We add <span class="m">0.1 × PE</span> to the token embeddings: the small scale keeps position from drowning out character identity in our tiny model.</div>
'''


@plate('plate_03-1b_clock_hands')
def _():
    W, H, x0, y0 = 880, 290, 190, 20
    tmax = 40
    waves = [(0, '#1155CC', 'k = 0, ω = 1'), (2, '#674EA7', 'k = 2, ω ≈ 0.215'), (5, '#38761D', 'k = 5, ω ≈ 0.0215')]
    out = ''
    for r, (kk, col, lab) in enumerate(waves):
        omega = 10000 ** (-2 * kk / 24)
        yc = y0 + 40 + r * 82
        pts = ' '.join(f'{x0 + t / tmax * (W - x0 - 20):.1f},{yc - 30 * math.sin(t * omega):.1f}' for t in [i / 10 for i in range(tmax * 10 + 1)])
        dots = ''.join(f'<circle cx="{x0 + t / tmax * (W - x0 - 20):.1f}" cy="{yc - 30 * math.sin(t * omega):.1f}" r="3.2" fill="{col}"/>' for t in range(tmax + 1))
        out += f'<line x1="{x0}" y1="{yc}" x2="{W-20}" y2="{yc}" stroke="#DDD"/>'
        out += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.5" opacity=".45"/>{dots}'
        out += f'<text x="{x0-14}" y="{yc-4}" text-anchor="end" font-family="Arial" font-size="15" fill="{col}" font-weight="bold">{["fast", "medium", "slow"][r]}</text>'
        out += f'<text x="{x0-14}" y="{yc+15}" text-anchor="end" font-family="Inc" font-size="13" fill="{col}">{lab}</text>'
    for t in range(0, tmax + 1, 10):
        x = x0 + t / tmax * (W - x0 - 20)
        out += f'<text x="{x:.1f}" y="{H-26}" text-anchor="middle" font-family="Inc" font-size="13" fill="#666">{t}</text>'
    svg = f'<svg width="{W}" height="{H+10}" viewBox="0 0 {W} {H+10}">{out}<text x="{(x0+W-20)/2}" y="{H-4}" text-anchor="middle" font-family="Arial" font-size="13" fill="#666">position t (dots); curves show sin(t ω) for three channel pairs of a width-24 model</text></svg>'
    return title('Like the Hands of a Clock') + f'''
<p>The first channel pair changes noticeably from one position to the next; later pairs change more and more slowly, telling apart positions far apart.
Together they give every position a unique fingerprint.</p>
<div style="text-align:center">{svg}</div>
<div class="row" style="margin-top:6px">
 <div class="card" style="margin:0;flex:1.25"><h3>Worked Example: D = 4</h3>Two frequencies, <span class="m">ω<sub>0</sub> = 1</span> and <span class="m">ω<sub>1</sub> = 10000<sup>−2/4</sup> = 0.01</span>.
 <div class="m" style="margin-top:6px">PE<sub>1</sub> = [sin 1, cos 1, sin 0.01, cos 0.01]<br>&nbsp;&nbsp;&nbsp;&nbsp;= [0.8415, 0.5403, 0.0100, 1.0000]</div>
 <div style="margin-top:4px">Position 0 is always <span class="m">[0, 1, 0, 1]</span>.</div></div>
 <div class="card" style="margin:0"><h3>Do we even need it here?</h3>ROT13 maps each character to a fixed partner regardless of its neighbors, so this task could be solved with no context. But position is essential for almost any real language task and part of every Transformer, so we include it. The supplied demo shows the shuffling problem directly.</div>
</div>
'''


# ---------------------------------------------------------------- Task 03-2
@plate('plate_03-2a_encoder_block')
def _():
    def box(x, w, label, sub, fill, line):
        return (f'<rect x="{x}" y="70" width="{w}" height="66" rx="8" fill="{fill}" stroke="{line}" stroke-width="2"/>'
                f'<text x="{x+w/2}" y="{98 if sub else 109}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="bold">{label}</text>'
                + (f'<text x="{x+w/2}" y="{121}" text-anchor="middle" font-family="Inc" font-size="13" fill="#333">{sub}</text>' if sub else ''))

    def plus(x):
        return f'<circle cx="{x}" cy="103" r="15" fill="#fff" stroke="#000" stroke-width="2"/><text x="{x}" y="110" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">+</text>'

    def arrow(x1, x2, y=103):
        return f'<line x1="{x1}" y1="{y}" x2="{x2-2}" y2="{y}" stroke="#000" stroke-width="2" marker-end="url(#ak)"/>'

    def skip(x1, x2):
        return f'<path d="M {x1} 103 V 38 H {x2} V {86}" fill="none" stroke="#38761D" stroke-width="2.5" stroke-dasharray="7 4" marker-end="url(#ag)"/>'
    svg = f'''<svg width="920" height="160" viewBox="0 0 920 160"><defs>
 <marker id="ak" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#000"/></marker>
 <marker id="ag" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#38761D"/></marker></defs>
 <text x="30" y="109" text-anchor="middle" font-family="Inc" font-size="22" font-weight="bold" fill="#CC0000">X</text>
 {arrow(46, 110)}{box(110, 170, 'Attention', 'one head', '#FFF2CC', '#FFD966')}{arrow(280, 345)}{plus(360)}
 {skip(70, 360)}
 {arrow(375, 435)}<text x="397" y="92" text-anchor="middle" font-family="Inc" font-size="18" font-weight="bold">U</text>
 {box(435, 250, 'FFN', '24 → 48, ReLU, 48 → 24', '#D9D2E9', '#8E7CC3')}{arrow(685, 745)}{plus(760)}
 {skip(420, 760)}
 {arrow(775, 840)}<text x="878" y="109" text-anchor="middle" font-family="Inc" font-size="20" font-weight="bold">X<tspan font-size="13" dy="5">next</tspan></text>
 <text x="215" y="30" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#38761D">residual: identity path</text>
 <text x="590" y="30" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#38761D">residual: identity path</text>
</svg>'''
    return title('One Encoder Block') + f'''
<p>A Transformer <b class="def">encoder block</b> pairs self-attention with a small feed-forward network (HW06–07), and wraps each in a <b class="def">residual connection</b>:</p>
<div class="eq" style="font-size:24px;margin-bottom:6px"><span>U = <span class="r">X</span> + Attention(<span class="r">X</span>)</span>, &nbsp;&nbsp; <span>X<sub>next</sub> = U + FFN(U)</span></div>
<div class="eq" style="font-size:20px;margin-top:0"><span>FFN(U) = ReLU(U<span class="b">W<sub>1</sub></span><sup>T</sup> + <span class="k2">b<sub>1</sub></span>)<span class="b">W<sub>2</sub></span><sup>T</sup> + <span class="k2">b<sub>2</sub></span></span></div>
<div style="text-align:center">{svg}</div>
<div class="row" style="margin-top:8px">
 <div class="callout ly fy" style="margin:0"><b>Attention</b> lets each position <b>gather</b> information from the others.</div>
 <div class="callout lp fp" style="margin:0"><b>The FFN</b> processes each position <b>separately</b> (the same small network everywhere), turning what was gathered into something useful. Its hidden layer is twice as wide.</div>
 <div class="callout lg fg" style="margin:0"><b>The residual</b> means each sublayer only learns a <b>change</b> to its input. And since <span class="m">d(X + f(X)) = 1 + f′(X)</span>, the "1" carries gradient through even when <span class="m">f′</span> is small.</div>
</div>
'''


@plate('plate_03-2b_our_encoder')
def _():
    flow = chain([('tokens', 'fk lk'), ('E + 0.1·PE', 'fr lr'), ('block 0', 'fy ly'), ('block 1', 'fy ly'), ('linear', 'fb lb'), ('logits<small>one per position</small>', 'fp lp')], size=16)
    rows = [('Embedding <span class="code">E</span>', '(41, 24)', '984'),
            ('Per block: <span class="code">Wq, Wk, Wv, Wo</span>', '(24, 24) each', '2,304'),
            ('Per block: <span class="code">W1, b1, W2, b2</span>', '(48, 24), (48,), (24, 48), (24,)', '2,376'),
            ('Output <span class="code">Wout, bout</span>', '(41, 24), (41,)', '1,025')]
    body = ''.join(f'<tr><td>{n}</td><td class="code">{s}</td><td class="m num">{c}</td></tr>' for n, s, c in rows)
    return title('Our Encoder, and the Backward Pass Through a Residual') + f'''
<div class="card" style="text-align:center">{flow}</div>
<table class="t" style="font-size:16px"><tr><th>Parameter</th><th>Shape</th><th class="num">Count (width 24)</th></tr>{body}
<tr><td colspan="2"><b>Total (2 blocks)</b></td><td class="m num"><b>11,369</b></td></tr></table>
<div class="row">
 <div class="callout lk fk" style="margin:0">A deliberately small teaching encoder: <b>one head</b> per block and no <b class="def">layer normalization</b> (a rescaling step real Transformers use to keep training stable). Like the RNN, one prediction per position, trained with the same <span class="code">masked_loss</span>.</div>
 <div class="callout lg fg" style="margin:0"><b>Backward through a residual:</b> since <span class="m">X<sub>next</sub> = U + FFN(U)</span>, the gradient reaching <span class="m">U</span> is the <b>sum of two paths</b>: the upstream gradient itself (identity) plus the path back through the FFN. Same at <span class="m">U = X + Attention(X)</span>.</div>
</div>
'''


# ---------------------------------------------------------------- Task 04-1
@plate('plate_04-1_training_the_encoder')
def _():
    steps = [('1', 'Forward', 'logits (B, T, 41)', 'fk lk'), ('2', 'Masked loss', 'loss and its logit gradient', 'fr lr'),
             ('3', 'Backward', 'through the whole encoder', 'fk lk'), ('4', 'Update', 'SGD, global grad norm clipped to 1', 'fb lb')]
    cyc = ''
    for i, (n, h, s, f) in enumerate(steps):
        if i:
            cyc += '<span class="arr">→</span>'
        cyc += f'<div class="box {f}" style="width:190px;padding:10px"><span class="n" style="font-size:14px">STEP {n}</span><br>{h}<small style="font-family:Arial;font-size:13px">{s}</small></div>'
    return title("The Same Training Loop as HW09") + f'''
<div class="flow" style="margin-bottom:14px">{cyc}</div>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><h3>Settings</h3><span class="m">25</span> epochs at learning rate <span class="m">0.4</span>. After each epoch, evaluate on validation and keep the parameters with the <b>lowest validation loss</b>.
 <p style="font-size:16px;color:#444;margin:8px 0 0">Accuracy saturates at 100% after a dozen epochs, so it can't break ties; loss keeps measuring how confident the correct predictions are.</p></div>
 <div class="card" style="margin:0"><h3>What should happen?</h3>Untrained, the loss starts near <span class="m">log 41 ≈ 3.71</span>. ROT13 is a simple, consistent mapping, so it should drop <b>well below 1</b> within five epochs.
 <p style="font-size:16px;margin:8px 0 0" class="r">Stalls near 3.7, or turns <span class="code">nan</span>? Recheck Tasks 01–03 before waiting out the run.</p></div>
</div>
<div class="callout ly fy"><b>Reading the attention.</b> The supplied plot shows both blocks' weights for one test phrase: row <span class="m">i</span> is output position <span class="m">i</span>, column <span class="m">j</span> is input position <span class="m">j</span>, darker means more weight.
Each panel has its <b>own color scale</b>, and the printed largest and smallest weights next to uniform (<span class="m">1/T</span>) tell a real pattern from a nearly flat one.</div>
<div class="callout lr fr" style="margin-bottom:0">The test set stays closed until Task 05.</div>
'''


# ---------------------------------------------------------------- Task 05-1
@plate('plate_05-1_pytorch_encoder')
def _():
    rows = [('One attention head', '<span class="code">nhead</span> heads, each in its own lower-dimensional subspace; results concatenated'),
            ('No layer normalization', 'layer normalization after each residual addition'),
            ('No projection biases', 'biases on every projection'),
            ('FFN width 48, ReLU', '<span class="code">dim_feedforward=48</span>, ReLU by default'),
            ('No dropout', '<span class="code">dropout=0.0</span> here, so the comparison is deterministic')]
    body = ''.join(f'<tr><td>{a_}</td><td style="background:#EEF3FC">{b_}</td></tr>' for a_, b_ in rows)
    return title("PyTorch's Transformer Encoder") + f'''
<p><span class="code">nn.TransformerEncoderLayer</span> packages everything in our block, plus the pieces we left out. <span class="code">nn.TransformerEncoder(layer, num_layers=2)</span> stacks two copies.</p>
<table class="t" style="font-size:16px"><tr><th>Our <span class="code">ScratchEncoder</span> block</th><th><span class="code">nn.TransformerEncoderLayer</span></th></tr>{body}</table>
<div class="row" style="margin-bottom:10px">
 <div class="card" style="margin:0"><h3>Two arguments need care</h3><span class="code">batch_first=True</span>: expect <span class="code">(B, T, D)</span> like our code (default is <span class="code">(T, B, D)</span>).<br>
 <span class="code">src_key_padding_mask=(tokens == PAD)</span>: masks padded <b>keys</b>, with the same <span class="code">True</span> = padding convention as our <span class="code">pad_mask</span>.</div>
 <div class="card" style="margin:0"><h3>Same input, different training</h3><span class="code">nn.Embedding</span> (<span class="code">padding_idx=PAD</span>) <span class="m">+ 0.1 × PE</span>, then an <span class="code">nn.Linear</span> output. Trained with <b class="def">Adam</b> at <span class="m">0.003</span> for <span class="m">15</span> epochs, gradients clipped as before.</div>
</div>
<div class="callout lr fr" style="margin-bottom:0"><b>Not a controlled comparison</b> like HW07's scratch-vs-PyTorch FFNs: two heads, normalization, biases and a different optimizer. Its job is to be an <b>independently built second opinion</b>.</div>
'''


# ---------------------------------------------------------------- Task 05-2
@plate('plate_05-2_three_decoders')
def _():
    cards = [('HW09 frozen RNN', 'loaded from <span class="code">rnn_reference.npz</span>: the reference run of HW09\'s scratch network (embedding 32, hidden 64), frozen', 'fb lb'),
             ('Scratch encoder', 'yours, from Tasks 01–04', 'fy ly'),
             ('PyTorch encoder', 'Task 05-1\'s library build', 'fp lp')]
    html = ''.join(f'<div class="callout {c}" style="margin:0"><b>{n}</b><br><span style="font-size:16px">{d}</span></div>' for n, d, c in cards)
    return title('Three Decoders, One Test Set') + f'''
<p>Every model is now fixed. We open the <b>test set</b> once and evaluate all three:</p>
<div class="row" style="margin-bottom:12px">{html}</div>
<div class="callout lk fk"><b>Baseline: the direct ROT13 rule.</b> A fixed, known substitution gets every character right by construction.
A learned model that only matches the rule has learned the rule, nothing more. The value of the learned decoders is what they teach us about sequence models, and that <b>two independent machines agree</b>.</div>
<div class="callout lr fr" style="margin-bottom:0"><b>Exact-sequence accuracy</b> is the stricter number: one wrong character anywhere, or a misplaced end-of-sequence token, makes the whole phrase count as wrong.</div>
'''


# ---------------------------------------------------------------- Task 06-1
@plate('plate_06-1_rot13_rule')
def _():
    letters = 'abcdefghijklmnopqrstuvwxyz'
    hot = {'g', 'u', 'r'}

    def cell(ch, i, cls):
        style = 'background:#FFF2CC;border-color:#FFD966' if i in (6, 17, 20) else ''
        return f'<div style="border:1px solid #DDD;{style};text-align:center;font-family:Inc;font-size:18px;padding:3px 0" class="{cls}">{ch}</div>'
    top = ''.join(cell(c, i, 'r') for i, c in enumerate(letters))
    bot = ''.join(cell(letters[(i + 13) % 26], i, 'g') for i in range(26))
    idx = ''.join(f'<div style="text-align:center;font-family:Inc;font-size:11px;color:#999">{i}</div>' for i in range(26))
    grid = f'<div style="display:grid;grid-template-columns:repeat(26,1fr);gap:0">{idx}{top}{bot}</div>'
    return title('The Rule Itself: ROT13') + f'''
<p>Replace each letter with the letter <b>13 places later</b>, wrapping from <span class="m">z</span> back to <span class="m">a</span>. The alphabet has 26 letters, so applying it twice returns the original:
ROT13 is <b>its own inverse</b>, and the same function both encodes and decodes. Spaces and digits are unchanged.</p>
<div class="card"><div style="display:grid;grid-template-columns:90px 1fr;align-items:center;gap:8px">
 <div class="note k" style="text-align:right;font-size:14px;line-height:2.1"><br>letter<br>ROT13</div>{grid}</div></div>
<div class="card"><h3>Worked Example: decoding <span class="code">gur</span></h3>
<div class="eqrow" style="display:block;font-size:20px;line-height:1.8">
 <span class="r">g</span> = 6 &nbsp;→&nbsp; 6 + 13 = 19 &nbsp;→&nbsp; <span class="g">t</span><br>
 <span class="r">u</span> = 20 &nbsp;→&nbsp; (20 + 13) mod 26 = 7 &nbsp;→&nbsp; <span class="g">h</span><br>
 <span class="r">r</span> = 17 &nbsp;→&nbsp; (17 + 13) mod 26 = 4 &nbsp;→&nbsp; <span class="g">e</span></div>
<p style="margin:8px 0 0">So <span class="code">gur</span> decodes to <span class="code">the</span>. A digit such as <span class="code">6</span> stays <span class="code">6</span>.</p></div>
'''


SPEC = {
    '### Task 01-1: Description': [
        (['A Different Way to Use Context', 'Queries, Keys, and Values', 'Scores, Scaling, and Softmax', 'Masking Padding', 'Worked Example'], [
            ('plate_01-1a_attention_context', 'Self-attention lets every position look at every position at once, instead of passing information along an RNN hidden state. Query q_i = x_i W_Q^T, key k_j = x_j W_K^T, value v_j = x_j W_V^T.'),
            ('plate_01-1b_scores_softmax_mask', 'Scores s_ij = q_i dot k_j divided by sqrt(d); softmax over the keys gives attention weights that sum to 1 in each row. Padded keys get score minus infinity, so exactly zero weight. No causal mask is needed here.'),
            ('plate_01-1c_worked_example', 'Worked example with d = 2: q = [1, 0], keys [1, 0], [0, 1], [1, 1]. Dot products 1, 0, 1; scaled 0.7071, 0, 0.7071; softmax 0.4011, 0.1978, 0.4011; with key 2 masked 0.6698, 0.3302, 0.'),
        ])],
    '### Task 01-2: Description': [
        (['From Weights to New Representations', 'Worked Example', 'Caching for the Backward Pass'], [
            ('plate_01-2a_attention_forward', 'C = A V and Y = C W_O^T. Q, K, V = X W_Q^T, X W_K^T, X W_V^T, all (B, T, D); A = softmax(Q K^T / sqrt(D)) is (B, T, T); Y has the input shape (B, T, D), so blocks can be stacked.'),
            ('plate_01-2b_worked_example', 'Worked example: c_0 = 0.4011 [1, 0] + 0.1978 [0, 2] + 0.4011 [3, 1] = [1.6044, 0.7967]. One head, no biases; PyTorch masks use True for positions that may be attended to. attention_forward returns a cache for the backward pass.'),
        ])],
    '### Task 02-1: Description': [
        (['Following the Gradient Backward', 'Worked Example'], [
            ('plate_02-1a_attention_backward', 'Backward through attention: dW_O = (dL/dY)^T C, dC = (dL/dY) W_O; dA = dC V^T, dV = A^T dC; dS from the softmax rule; dQ = dS K / sqrt(d), dK = dS^T Q / sqrt(d); dW_Q = dQ^T X; dX = dQ W_Q + dK W_K + dV W_V, summing all three paths.'),
            ('plate_02-1b_softmax_rule', 'Softmax rule per row: dS_ij = A_ij (dA_ij minus the sum over k of dA_ik A_ik). Example: a = [0.7, 0.2, 0.1], dA = [1, 0, 0], weighted average 0.7, dS = [0.21, -0.14, -0.07], which sums to zero. Masked keys get zero gradient.'),
        ])],
    '### Task 03-1: Description': [
        (['Attention Has No Sense of Order', 'Sinusoidal Positional Encodings', 'Worked Example', 'Do We Even Need It Here?'], [
            ('plate_03-1a_positional_encoding', 'Attention treats its input as a set, so a positional encoding is added to each embedding: PE_t,2k = sin(t omega_k), PE_t,2k+1 = cos(t omega_k), omega_k = 10000 to the power -2k/D. We add 0.1 times PE.'),
            ('plate_03-1b_clock_hands', 'Early channel pairs oscillate quickly and later ones slowly, giving every position a unique fingerprint. Worked example with D = 4: PE_1 = [sin 1, cos 1, sin 0.01, cos 0.01] = [0.8415, 0.5403, 0.0100, 1.0000]; position 0 is [0, 1, 0, 1].'),
        ])],
    '### Task 03-2: Description': [
        (['One Encoder Block', 'Our Encoder', 'The Backward Pass Through a Residual'], [
            ('plate_03-2a_encoder_block', 'Encoder block: U = X + Attention(X), X_next = U + FFN(U), FFN(U) = ReLU(U W_1^T + b_1) W_2^T + b_2. Residuals pass gradient through because d(X + f(X)) = 1 + f prime.'),
            ('plate_03-2b_our_encoder', 'Our encoder: tokens, embedding plus 0.1 PE, two blocks, linear output. Parameters: E 984, per block attention 2,304 and FFN 2,376, output 1,025, total 11,369. Backward through a residual adds the identity path to the sublayer path.'),
        ])],
    '### Task 04-1: Description': [
        (['The Same Training Loop as HW09', 'Worked Example: What Should Happen?', 'Reading the Attention'], [
            ('plate_04-1_training_the_encoder', 'Training loop: forward, masked loss, backward, clipped SGD update. 25 epochs at learning rate 0.4, keeping the lowest validation loss. Loss starts near log 41 = 3.71 and should fall well below 1 within five epochs. The attention plot shows each block on its own color scale.'),
        ])],
    '### Task 05-1: Description': [
        (["PyTorch's Transformer Encoder"], [
            ('plate_05-1_pytorch_encoder', 'nn.TransformerEncoderLayer adds multiple heads, layer normalization, biases, FFN width 48 with ReLU, dropout 0. Use batch_first=True and src_key_padding_mask = (tokens == PAD). Trained with Adam at 0.003 for 15 epochs. Not a controlled comparison.'),
        ])],
    '### Task 05-2: Description': [
        (['Three Decoders, One Test Set'], [
            ('plate_05-2_three_decoders', 'Three decoders on the test set: the frozen HW09 RNN, the scratch encoder and the PyTorch encoder, with the direct ROT13 rule as a baseline. Exact-sequence accuracy counts a phrase wrong if any character is wrong.'),
        ])],
    '### Task 06-1: Description': [
        (['The Rule Itself', 'Worked Example'], [
            ('plate_06-1_rot13_rule', 'ROT13 shifts each letter 13 places, wrapping from z to a, and is its own inverse. g = 6 becomes 19 = t; u = 20 becomes 7 = h; r = 17 becomes 4 = e, so gur decodes to the. Digits are unchanged.'),
        ])],
}
EXTRA = {
    '### Task 06-1: Description': ("The supplied cell applies the rule to the note, then reads the note with all three learned decoders and checks whether each reading "
                                   "agrees with the rule exactly. There's no code to write: the question is whether three machines built three different ways tell Caulfield the same thing."),
}

if __name__ == '__main__':
    main('10', PLATES, SPEC, EXTRA)
