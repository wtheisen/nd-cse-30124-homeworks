"""HW08 explanation plates (lecture-style PNGs) and the notebook swap.

Nov 8 2026 rebuild on the SP26 classes: 02-2b, 03-1a/b, 04-1, 04-2a/b/c, 05-1a/b rewritten for
Rot13Dataset / EmbeddingLayer / RecurrentBlock / RecurrentNeuralNetwork / CharRNN; 05-2a/b moved out (metrics are HW09).

Nov 2026 polish: added 01-1a (cosine primer), 01-1b, 01-1c (five ciphers, Nov 8; Two Quick Tests and the 01-2 fingerprint plate were cut), 02-2b, 03-1a, 03-2b, 04-2b, 05-1b, 05-2b and lettered the
existing plates to match HW07. The notebooks already had plates, so the new ones were
wired in with _wire08.py (--apply only touches cells without plates).

    python plates08.py            # render imgs/homework08/plates/*.png
    python plates08.py --apply    # swap them into homework08.ipynb and homework08_solutions.ipynb
"""
import math

from plates_common import *  # noqa: F401,F403

PLATES, plate = plate_registry()


# ---------------------------------------------------------------- Task 01-1
@plate('plate_01-1a_cosine_similarity')
def _():
    S, X0, Y0 = 52, 50, 250          # 52 px per count; origin at (50, 250)
    P = lambda x, y: (X0 + x * S, Y0 - y * S)

    def arrow(x, y, col, lab, dx=8, dy=-8):
        ex, ey = P(x, y)
        return (f'<line x1="{X0}" y1="{Y0}" x2="{ex}" y2="{ey}" stroke="{col}" stroke-width="4" marker-end="url(#h{col[1:]})"/>'
                f'<text x="{ex+dx}" y="{ey+dy}" font-family="Inc" font-size="17" font-weight="bold" fill="{col}">{lab}</text>')
    marks = ''.join(f'<marker id="h{c[1:]}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                    for c in ('#1155CC', '#CC0000', '#674EA7'))
    grid = ''.join(f'<line x1="{X0+i*S}" y1="{Y0}" x2="{X0+i*S}" y2="{Y0-4.4*S}" stroke="#EEE"/><line x1="{X0}" y1="{Y0-i*S}" x2="{X0+4.4*S}" y2="{Y0-i*S}" stroke="#EEE"/>'
                   for i in range(1, 5))
    ticks = ''.join(f'<text x="{X0+i*S}" y="{Y0+20}" text-anchor="middle" font-family="Inc" font-size="14" fill="#666">{i}</text>'
                    f'<text x="{X0-12}" y="{Y0-i*S+5}" text-anchor="middle" font-family="Inc" font-size="14" fill="#666">{i}</text>' for i in range(1, 5))
    a0, a1 = math.atan2(1, 3), math.atan2(2, 1)
    r = 46
    arc = (f'<path d="M{X0+r*math.cos(a0):.1f},{Y0-r*math.sin(a0):.1f} A{r},{r} 0 0 0 {X0+r*math.cos(a1):.1f},{Y0-r*math.sin(a1):.1f}" fill="none" stroke="#666" stroke-width="2"/>'
           f'<text x="{X0+52}" y="{Y0-32}" font-family="Inc" font-size="17" fill="#666">θ</text>')
    svg = (f'<svg width="300" height="290" viewBox="0 0 300 290"><defs>{marks}</defs>{grid}'
           f'<line x1="{X0}" y1="{Y0}" x2="{X0+4.5*S}" y2="{Y0}" stroke="#000" stroke-width="1.5"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y0-4.5*S}" stroke="#000" stroke-width="1.5"/>'
           f'{ticks}<text x="{X0+4.5*S}" y="{Y0+38}" text-anchor="end" font-family="Arial" font-size="14" fill="#666">count of x</text>'
           f'<text x="{X0+6}" y="{Y0-4.5*S+4}" font-family="Arial" font-size="14" fill="#666">count of y</text>'
           f"{arrow(2, 4, '#CC0000', 'b')}{arrow(1, 2, '#1155CC', 'a', -24, 0)}{arrow(3, 1, '#674EA7', 'c', 10, 4)}{arc}</svg>")
    return title('Comparing Directions: Cosine Similarity') + f'''
<p>Once a text is a list of counts, it's a <b class="def">vector</b>, and comparing two texts means comparing two vectors. <b class="def">Cosine similarity</b> compares their <b>directions</b> (which mix of items each has) and ignores their <b>lengths</b> (how long each text is).</p>
<div class="row" style="align-items:center;margin-bottom:10px">
 <div style="flex:0 0 300px">{svg}</div>
 <div>
  <div class="eq" style="font-size:26px;margin:0 0 6px">cos θ = <span class="frac"><span><span class="b">a</span>·<span class="r">b</span></span><span>‖<span class="b">a</span>‖ ‖<span class="r">b</span>‖</span></span></div>
  <p style="font-size:16px;text-align:center;margin:0 0 10px" class="k">θ is the angle between two arrows (drawn here for <span class="b">a</span> and <span class="p">c</span>; <span class="b">a</span> and <span class="r">b</span> point the same way, so their θ is 0)</p>
  <table class="t" style="font-size:16px;margin:0"><tr><th>Piece</th><th>Meaning</th><th>For <span class="b">a = [1, 2]</span>, <span class="r">b = [2, 4]</span></th></tr>
  <tr><td class="m"><span class="b">a</span>·<span class="r">b</span></td><td>dot product: multiply matching entries, add them up</td><td class="m">1·2 + 2·4 = 10</td></tr>
  <tr><td class="m">‖<span class="b">a</span>‖</td><td>length: square root of the sum of squares</td><td class="m">‖<span class="b">a</span>‖ = √5, ‖<span class="r">b</span>‖ = √20</td></tr>
  <tr><td class="m">cos θ</td><td>dot product over both lengths</td><td class="m">10 / √100 = <span class="g">1</span></td></tr></table>
 </div>
</div>
<div class="row" style="margin-bottom:10px">
 <div class="card" style="margin:0"><h3>Same direction: <span class="g">1</span></h3><span class="m b">a</span> and <span class="m r">b</span> have the same proportions; <span class="m r">b</span> is just twice as long, like a text and the same text written twice.</div>
 <div class="card" style="margin:0"><h3>45° apart: <span class="p">0.707</span></h3><span class="m"><span class="b">a</span>·<span class="p">c</span> = 1·3 + 2·1 = 5</span> for <span class="m p">c = [3, 1]</span>, over <span class="m">√5 · √10 = √50</span>. Some overlap, a different mix.</div>
 <div class="card" style="margin:0"><h3>Nothing shared: <span class="r">0</span></h3><span class="m">[1, 0]</span> and <span class="m">[0, 1]</span> are at 90°: every product in the dot product is 0.</div>
</div>
<div class="callout lg fg" style="margin-bottom:0"><b>Why not just measure the distance?</b> <span class="code">hello</span> and <span class="code">hellohello</span> have counts <span class="m">[1, 1, 2, 1]</span> and <span class="m">[2, 2, 4, 2]</span>: about 2.65 apart, yet their cosine is exactly 1. Counts are never negative, so for bags the cosine always lands between 0 and 1.</div>
'''


# ---------------------------------------------------------------- Task 01-1
@plate('plate_01-1c_five_candidate_ciphers')
def _():
    rows = [('<span class="code">Caesar(3)</span>', 'every letter moves 3 places later', 'KHOOR', True),
            ('<span class="code">Caesar(13)</span>: ROT13', 'every letter moves 13 places later; applying it twice undoes it', 'URYYB', True),
            ('<span class="code">Vigenere(\'clue\')</span>', 'the shift changes letter by letter, cycling through the key: +2, +11, +20, +4, …', 'JPFPQ', False),
            ('<span class="code">Affine(5, 8)</span>', 'letter number x becomes (5x + 8) mod 26', 'RCLLA', True),
            ('<span class="code">SimpleSubstitution</span>', 'a fixed scrambled alphabet: a → q, b → w, c → e, …', 'ITSSG', True)]
    keep = '<span class="g">renamed</span>: each letter always becomes the same letter'
    blur = '<span class="r">blurred</span>: the two <span class="code lit">l</span>s became <span class="code lit">F</span> and <span class="code lit">P</span>'
    body = ''.join(f'<tr><td><b>{n}</b></td><td style="font-size:15px">{r}</td><td class="code" style="font-size:17px">{e}</td>'
                   f'<td style="font-size:15px">{keep if k else blur}</td></tr>' for n, r, e, k in rows)
    flow = chain([("<span class='code'>SAMPLE</span><small>a paragraph of English</small>", 'fk lk'),
                  ('encode<small>with one candidate</small>', 'fb lb'),
                  ('compare bags<small>with the note</small>', 'fp lp'),
                  ('score<small>BoW + BoC</small>', 'fg lg')], size=15)
    return title('Five Candidate Ciphers') + f'''
<p>The cipher unit has five disguises on file for this kind of case, all from the <span class="code">pycipher</span> library. <span class="code">encipher</span> returns <b>capitals</b> and <b>drops</b> spaces, digits and punctuation.</p>
<table class="t" style="font-size:16px"><tr><th>Cipher</th><th>Rule</th><th><span class="code">hello</span> →</th><th>Letter counts</th></tr>{body}</table>
<div class="card"><h3>The test: disguise a known English text with each candidate</h3>
<div style="margin:6px 0 10px">{flow}</div>
<span style="font-size:17px">Encode the sample with the <b>right</b> cipher and it gets the same renaming as the note: each letter's count lands on the same disguised letter (English's many <span class="code lit">e</span>s become <span class="code lit">r</span>s under ROT13, as in the note). A wrong cipher renames differently, so the character bags line up worse.</span></div>
<div class="callout lr fr" style="margin-bottom:0"><b>The keys are fixed.</b> We test one shift, one Vigenère key, one affine rule and one scrambled alphabet. If the note used Caesar with shift 7, or a different scramble, none of these would match well. A low score rules out <b>this key</b>, not the whole family.</div>
'''


# ---------------------------------------------------------------- Task 02-1
@plate('plate_02-1_characters_become_ids')
def _():
    groups = [('0', '&lt;PAD&gt;', 'fk lk', 'padding'), ('1', '&lt;EOS&gt;', 'fr lr', 'end of phrase'),
              ('2', '&lt;UNK&gt;', 'fk lk', 'unknown'), ('3', '&lt;SOS&gt;', 'fk lk', 'unused'),
              ('4', "' '", 'fy ly', 'space'), ('5–30', 'a … z', 'fb lb', 'letters'), ('31–40', '0 … 9', 'fp lp', 'digits')]
    strip = ''.join(f'<div class="box {f}" style="min-width:0;flex:{3 if "–" in i else 1};padding:8px 4px;font-size:15px"><span class="m" style="font-size:17px">{t}</span><small style="font-family:Arial">ID {i}</small><small style="font-family:Arial;color:#666">{d}</small></div>'
                    for i, t, f, d in groups)
    chars = [('h', 12, 'fb lb'), ('e', 9, 'fb lb'), ('l', 16, 'fb lb'), ('l', 16, 'fb lb'), ('o', 19, 'fb lb'), ('␣', 4, 'fy ly'), ('6', 37, 'fp lp'), ('9', 40, 'fp lp'), ('EOS', 1, 'fr lr')]
    cols = ''.join(f'<div style="text-align:center"><div class="m" style="font-size:20px;height:30px">{c}</div><div class="box {f}" style="min-width:56px;padding:6px;font-family:Inc;font-size:20px">{i}</div></div>' for c, i, f in chars)
    return title('Characters Become IDs') + f'''
<p>A network needs numbers, so every character gets an integer <b class="def">token ID</b> from a fixed <b class="def">vocabulary</b> of <span class="m">V = 41</span> tokens:</p>
<div style="display:flex;gap:6px;margin-bottom:16px">{strip}</div>
<div class="row" style="margin-bottom:12px">
 <div class="callout lr fr" style="margin:0">A token ID is a <b>category label</b>, like HW06's class IDs: <span class="code">z</span> (ID 30) is not "six times" <span class="code">a</span> (ID 5). Task 03 deals with that.</div>
 <div class="callout lb fb" style="margin:0"><b>Why <span class="code">&lt;EOS&gt;</span>?</b> After padding, it marks where the real text stops. It's a genuine target too: the model must learn to predict "the phrase ends here".</div>
</div>
<div class="card" style="margin:0"><h3>Worked Example: <span class="code">encode('hello 69')</span></h3>
<div style="display:flex;gap:8px;justify-content:center;margin:6px 0 10px">{cols}</div>
<b><span class="code">decode</span></b> reverses it: <b>stops</b> at the first <span class="code">&lt;EOS&gt;</span>, <b>skips</b> <span class="code">&lt;PAD&gt;</span>, and shows any other special token as <span class="code">?</span>.</div>
'''


# ---------------------------------------------------------------- Task 02-2
@plate('plate_02-2a_padding')
def _():
    def grid(rows, lab):
        out = ''
        for r in rows:
            out += '<div style="display:flex">' + ''.join(
                f'<span class="{"fk lk" if v == 0 else ("fr lr" if v == 1 else "fb lb")}" style="font-family:Inc;font-size:20px;width:52px;text-align:center;padding:6px 0;border:2px solid;margin:-1px 0 0 -2px;{"color:#999" if v == 0 else ""}">{v}</span>'
                for v in r) + '</div>'
        return f'<div style="text-align:center"><div class="lab" style="margin-bottom:6px">{lab}</div><div style="display:inline-block">{out}</div></div>'
    src = [[25, 22, 29, 29, 6, 1], [37, 40, 1, 0, 0, 0]]
    tgt = [[12, 9, 16, 16, 19, 1], [37, 40, 1, 0, 0, 0]]
    return title('Batches of Different Lengths') + f'''
<p>HW06's images were all 28 × 28, but phrases have different lengths and a tensor must be rectangular. <b class="def">Padding</b> appends
<span class="code">&lt;PAD&gt;</span> (ID 0) to every shorter sequence until it is as long as the longest one: a batch of <span class="m">B</span> phrases becomes a <span class="code">(B, T)</span> tensor.</p>
<h3>Worked Example: (<span class="code">uryyb</span> → <span class="code">hello</span>) and (<span class="code">69</span> → <span class="code">69</span>)</h3>
<div style="display:flex;justify-content:space-around;align-items:center;margin:8px 0 14px">
 {grid(src, 'sources (2, 6)')}<span class="arr" style="font-size:30px">⇢</span>{grid(tgt, 'targets (2, 6)')}
</div>
<p style="font-size:16px;text-align:center;margin:-4px 0 14px" class="k">Digits are unchanged by ROT13, so the second pair's source and target are identical. <span style="color:#CC0000;font-weight:bold">1</span> is <span class="code">&lt;EOS&gt;</span>, gray <span class="code">0</span> is padding.</p>
<div class="row">
 <div class="callout lg fg" style="margin:0"><b>Aligned:</b> ROT13 swaps one character for one character, so a phrase and its translation always have the same length. Position <span class="m">t</span> of the source lines up with position <span class="m">t</span> of the target, so the model can make <b>one prediction per position</b>.</div>
 <div class="callout lr fr" style="margin:0"><b>Padding isn't evidence.</b> No credit for "predicting" it: every loss and accuracy we compute will <b>ignore</b> <span class="code">&lt;PAD&gt;</span> positions (Task 05-2).</div>
</div>
'''


# ---------------------------------------------------------------- Task 03-1
@plate('plate_03-1b_embeddings')
def _():
    W = mat('b', 3, [0, 1, 3, 0, 2, 4], hl=(2, 5))
    return title('From IDs to Vectors: An Embedding Layer') + f'''
<p>An <b class="def">embedding layer</b> is a <span class="code">LinearLayer</span> from <span class="m">V</span> inputs to <span class="m">E</span> outputs, applied to one-hot vectors, with no bias.
Its weight <span class="m b">W</span> has shape <span class="code">(E, V)</span>: <b>one column per token</b>. Multiplying by a one-hot vector picks out that token's column:</p>
<div class="eq">onehot(<span class="r">i</span>) <span class="b">W</span><sup>T</sup> = column <span class="r">i</span> of <span class="b">W</span></div>
<div class="card"><h3>Worked Example: V = 3 tokens, E = 2</h3>
<div class="eqrow" style="justify-content:center;gap:10px"><span class="b">W</span> = {W}
 <span class="note" style="margin:0 10px 0 26px">tokens</span> <span class="r">[2, 1, 2]</span> → {mat('r', 2, [3, 4, 1, 2, 3, 4])}</div>
<p style="font-size:16px;margin:8px 0 0" class="k">Token 2's vector is column 2 of W, <span class="m">[3, 4]</span>. Repeated IDs pick the same column again.</p></div>
<div class="row">
 <div class="card" style="margin:0"><h3>Whole batches at once</h3><span class="code">(B, T)</span> IDs → <span class="code">F.one_hot</span> → <span class="code">(B, T, V)</span> → <span class="code">@ W.T</span> → <span class="code">(B, T, E)</span>. Our network uses <span class="m">E = 32</span>.</div>
 <div class="card" style="margin:0"><h3>Learned in HW09</h3>The columns start random. Training adjusts them so tokens that should be treated alike get similar vectors.</div>
 <div class="card fk" style="margin:0;background:#F6F6F6"><h3>The inherited bias</h3><span class="code">EmbeddingLayer</span> inherits a bias <span class="code">b</span> from <span class="code">LinearLayer</span>, but its <span class="code">forward</span> never adds it.</div>
</div>
'''


# ---------------------------------------------------------------- Task 03-2
@plate('plate_03-2a_tanh')
def _():
    pts = ' '.join(f'{40+(a+4)*45:.1f},{110-80*math.tanh(a):.1f}' for a in [x / 10 for x in range(-40, 41)])
    sig = ' '.join(f'{40+(a+4)*45:.1f},{110-80*(1/(1+math.exp(-a))):.1f}' for a in [x / 10 for x in range(-40, 41)])
    svg = f'''<svg width="420" height="220" viewBox="0 0 420 220">
 <line x1="40" y1="30" x2="400" y2="30" stroke="#999" stroke-dasharray="4 4"/><line x1="40" y1="190" x2="400" y2="190" stroke="#999" stroke-dasharray="4 4"/>
 <line x1="40" y1="110" x2="400" y2="110" stroke="#000" stroke-width="1.5"/><line x1="220" y1="15" x2="220" y2="205" stroke="#000" stroke-width="1.5"/>
 <polyline points="{sig}" fill="none" stroke="#CCCCCC" stroke-width="3"/>
 <polyline points="{pts}" fill="none" stroke="#DCB439" stroke-width="5"/>
 <text x="404" y="34" font-family="Inc" font-size="15">1</text><text x="404" y="194" font-family="Inc" font-size="15">−1</text>
 <text x="240" y="150" font-family="Arial" font-size="15" font-weight="bold" fill="#DCB439">tanh</text>
 <text x="300" y="82" font-family="Arial" font-size="14" fill="#999">sigmoid (HW03)</text>
 <text x="396" y="128" font-family="Inc" font-size="15">a</text>
</svg>'''
    return title('A Bounded Activation: tanh') + f'''
<div class="row" style="align-items:center">
 <div style="flex:1.1"><p>The classic recurrent network uses the <b class="def">hyperbolic tangent</b>. It's S-shaped like HW03's sigmoid but centered on zero:
 it maps any number into <span class="m">(−1, 1)</span>, with <span class="m">tanh(0) = 0</span>.</p>
 <div class="eq" style="font-size:24px">tanh(a) = <span class="frac"><span>e<sup>a</sup> − e<sup>−a</sup></span><span>e<sup>a</sup> + e<sup>−a</sup></span></span></div>
 <p style="font-size:17px">Slope, needed next week: <span class="m">d tanh(a)/da = 1 − tanh(a)<sup>2</sup></span></p></div>
 <div style="flex:0 0 420px">{svg}</div>
</div>
<div class="row">
 <div class="card" style="margin:0"><h3>Worked Example</h3>
 <div class="m" style="font-size:19px;line-height:1.9">tanh(1) = <span class="frac"><span>2.718 − 0.368</span><span>2.718 + 0.368</span></span> = <span class="frac"><span>2.350</span><span>3.086</span></span> = <span class="y">0.7616</span></div></div>
 <div class="card" style="margin:0"><h3>Odd and bounded</h3>
 <div class="m" style="font-size:19px;line-height:1.9">tanh(−1) = <span class="y">−0.7616</span><br>tanh(3) = <span class="y">0.9951</span> <span class="note k">already close to the ceiling</span></div></div>
</div>
'''


# ---------------------------------------------------------------- Task 04-1
@plate('plate_04-1_rnn_step')
def _():
    rows = [('<span class="r">x<sub>t</sub></span>', 'embeddings of the characters at position t, one row per phrase', '(B, 32)'),
            ('<span class="y">h<sub>t−1</sub></span>', 'previous hidden states', '(B, 64)'),
            ('<span class="b">W</span>', 'turns the current input into hidden-state space', '(64, 32)'),
            ('<span class="b">U</span>', 'turns the previous hidden state into hidden-state space', '(64, 64)'),
            ('<span class="k2">b</span>', 'hidden bias', '(64,)'),
            ('<span class="y">h<sub>t</sub></span>', 'new hidden states', '(B, 64)')]
    body = ''.join(f'<tr><td class="m">{a}</td><td>{b}</td><td class="code">{c}</td></tr>' for a, b, c in rows)
    return title('A Network with a Memory') + f'''
<p>Text arrives in order, and what a character means can depend on what came before. A <b class="def">recurrent neural network</b> reads one position at a time and keeps a <b class="def">hidden state</b> <span class="m y">h<sub>t</sub></span>, a vector summarizing everything read so far. One step of <span class="code">RecurrentBlock.forward</span>:</p>
<div class="eq"><span class="p">z<sub>t</sub></span> = <span class="r">x<sub>t</sub></span> <span class="b">W</span><sup>T</sup> + <span class="y">h<sub>t−1</sub></span> <span class="b">U</span><sup>T</sup> + <span class="k2">b</span>, &nbsp; <span class="y">h<sub>t</sub></span> = tanh(<span class="p">z<sub>t</sub></span>)</div>
<div class="row" style="align-items:flex-start">
 <div style="flex:1.35"><table class="t" style="font-size:16px;margin:0"><tr><th>Symbol</th><th>Meaning</th><th>Shape</th></tr>{body}</table>
 <p style="font-size:15px;margin:8px 0 0" class="k">Both products are a linear layer's "outputs × inputs" weights. The first step starts from <span class="m">h<sub>0</sub> = 0</span>.</p></div>
 <div class="card" style="margin:0"><h3>Worked Example</h3>
 <div style="font-size:15px;margin-bottom:6px"><span class="m r">x<sub>t</sub> = [1, 0]</span>, <span class="m y">h<sub>t−1</sub> = [0.5, −0.5]</span>,<br><span class="m b">W = I</span>, <span class="m b">U = 0.5 I</span>, <span class="m k2">b = [0, 0.1]</span></div>
 <div class="m" style="font-size:18px;line-height:1.8">
 <span class="r">x<sub>t</sub></span><span class="b">W</span><sup>T</sup> = [1, 0]<br>
 <span class="y">h<sub>t−1</sub></span><span class="b">U</span><sup>T</sup> = [0.25, −0.25]<br>
 <span class="p">z<sub>t</sub> = [1.25, −0.15]</span> <span class="note k">sum plus bias</span><br>
 <span class="y">h<sub>t</sub></span> = tanh(<span class="p">z<sub>t</sub></span>) = <span class="y">[0.8483, −0.1489]</span></div></div>
</div>
'''


# ---------------------------------------------------------------- Task 04-2
@plate('plate_04-2a_unrolling')
def _():
    chars = ['u', 'r', 'y', 'y', 'b', 'EOS']
    cols = ''
    for i, c in enumerate(chars):
        cols += f'''<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
          <div class="box fp lp" style="min-width:58px;padding:4px;font-size:14px">logits</div><span class="arr" style="font-size:18px;padding:0">↑</span>
          <div class="box fy ly" style="min-width:58px;padding:8px 4px;font-size:16px">h<sub>{i+1}</sub></div><span class="arr" style="font-size:18px;padding:0">↑</span>
          <div class="box fr lr" style="min-width:58px;padding:4px;font-size:15px;font-family:Inc">{c}</div></div>'''
        if i < len(chars) - 1:
            cols += '<span class="arr y" style="font-size:22px;padding:0 2px;margin-top:6px">→</span>'
    return title('Unrolling Over the Whole Sequence') + f'''
<p>Apply the same step at every position, passing each hidden state on to the next. Drawn out across time, the loop looks like a very deep network whose layers all <b>share the same weights</b>.</p>
<div style="display:flex;align-items:center;justify-content:center;margin:6px 0 4px"><div class="box fy ly" style="min-width:0;padding:8px;font-size:16px;margin-top:6px">h<sub>0</sub>=0</div><span class="arr y" style="font-size:22px;margin-top:6px">→</span>{cols}</div>
<p style="text-align:center;font-size:15px" class="k">every column uses the same <span class="m b">W</span> and <span class="m b">U</span> (and the same output layer); only the hidden state changes</p>
<div class="row" style="margin-bottom:10px">
 <div class="card" style="margin:0"><h3>One prediction per position</h3>The <span class="code">RecurrentBlock</span> returns every hidden state, <span class="code">(B, T, 64)</span>. The output <span class="code">LinearLayer</span> then turns each one into 41 logits:
 <span class="code">(B, T)</span> source IDs → <span class="code">(B, T, 41)</span> logits. The target at position <span class="m">t</span> is the translation of source character <span class="m">t</span>.</div>
 <div class="card" style="margin:0"><h3>The loop runs over time</h3>At each <span class="code">timestep</span>, <span class="code">X[:, timestep, :]</span> holds that character's embedding for <b>every</b> phrase, shape <span class="code">(B, 32)</span>: the whole batch moves through time together.</div>
</div>
<div class="callout lp fp" style="margin-bottom:0"><b>Caching for next week:</b> HW09's backpropagation through time needs every step's input, new hidden state and previous hidden state.
<span class="code">self.hidden_states</span> keeps a <span class="code">Grad_Info</span> for each step after the zero start: <span class="m">T + 1</span> entries in all.</div>
'''


@plate('plate_04-2c_parameters')
def _():
    rows = [('EmbeddingLayer.W', '32 × 41', '1,312'), ('RecurrentBlock.W', '64 × 32', '2,048'), ('RecurrentBlock.U', '64 × 64', '4,096'),
            ('RecurrentBlock.b', '64', '64'), ('output LinearLayer.W', '41 × 64', '2,624'), ('output LinearLayer.b', '41', '41')]
    body = ''.join(f'<tr><td><span class="code" style="font-weight:bold">{a}</span></td><td class="m">{c}</td><td class="m num">{d}</td></tr>' for a, c, d in rows)
    return title('Counting Parameters') + f'''
<div class="row" style="align-items:center">
 <div style="flex:1.2"><table class="t" style="margin:0"><tr><th>Parameter</th><th>Shape</th><th class="num">Count</th></tr>{body}
 <tr><td><b>Total used by the forward pass</b></td><td></td><td class="m num"><b>10,185</b></td></tr></table>
 <p style="font-size:15px;margin:8px 0 0" class="k">The embedding layer also carries an unused bias of 32, inherited from <span class="code">LinearLayer</span>.</p></div>
 <div>
  <div class="callout lp fp" style="margin-top:0">That's tiny next to HW06's <span class="m">239,919</span>.</div>
  <div class="callout lg fg">The count <b>doesn't depend on sequence length</b>: a 6-character phrase and the 263-character note use the same 10,185 numbers.</div>
 </div>
</div>
'''


# ---------------------------------------------------------------- Task 05-1
@plate('plate_05-1a_library_version')
def _():
    rows = [('<span class="code">EmbeddingLayer</span>', 'nn.Embedding(vocab_size, embed_size)', '<span class="code">.weight</span> is <span class="code">(V, 32)</span>: one <b>row</b> per token, our <span class="code">W</span> transposed'),
            ('<span class="code">RecurrentBlock</span>', 'nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)', 'tanh by default; returns all hidden states'),
            ('<span class="code">W</span>, <span class="code">U</span>', '.weight_ih_l0, .weight_hh_l0', '"input-to-hidden" and "hidden-to-hidden", layer 0'),
            ('<span class="code">b</span>', '.bias_ih_l0 + .bias_hh_l0', 'PyTorch adds <b>two</b> biases'),
            ('output <span class="code">LinearLayer</span>', 'nn.Linear(hidden_size, vocab_size)', '<span class="code">.weight</span> <span class="code">(V, 64)</span>, <span class="code">.bias</span> <span class="code">(V,)</span>')]
    body = ''.join(f'<tr><td>{a}</td><td class="code" style="background:#EEF3FC;font-size:14px;white-space:normal">{b}</td><td style="font-size:15px">{c}</td></tr>' for a, b, c in rows)
    return title('The Library Version') + f'''
<table class="t"><tr><th>Our scratch version</th><th>PyTorch (<span class="code">CharRNN</span>)</th><th>Notes</th></tr>{body}</table>
<div class="row">
 <div class="card" style="margin:0"><h3><span class="code">batch_first=True</span></h3>Makes <span class="code">nn.RNN</span> accept <span class="code">(B, T, 32)</span> like ours (its default is <span class="code">(T, B, 32)</span>).
 It returns the hidden states at every position, <span class="code">(B, T, 64)</span>, and the final state. We only need the first.</div>
 <div class="callout lr fr" style="margin:0"><b>The two-bias detail.</b> PyTorch computes
 <div class="m" style="margin:6px 0;font-size:16px">tanh(<span class="r">x</span> <span class="b">W<sub>ih</sub></span><sup>T</sup> + <span class="k2">b<sub>ih</sub></span> + <span class="y">h</span> <span class="b">W<sub>hh</sub></span><sup>T</sup> + <span class="k2">b<sub>hh</sub></span>)</div>
 Two biases added together do what our one does, so the supplied check copies our <span class="code">b</span> into <span class="code">bias_ih_l0</span> and sets <span class="code">bias_hh_l0</span> to zero.</div>
</div>
<div class="callout lb fb" style="margin-bottom:0">As in HW06, matching architecture isn't enough: the supplied check copies every parameter from the scratch model before comparing outputs.</div>
'''


# ---------------------------------------------------------------- Task 05-2
# ---------------------------------------------------------------- code blocks (as in plates07.py)
import html as _html
import re as _re


def code(src, numbers=True, size=13.5):
    """A code block in the book's style: Consolas, gutter, keywords blue, numbers/strings red, comments green."""
    tok = _re.compile(r"(#.*$)|('[^']*')|((?<![\w.])\d+(?![\w]))|\b(class|def|return|for|in|import|from|as|True|False|None)\b")

    def hl(ln):
        out, last = [], 0
        for m in tok.finditer(ln):
            out.append(_html.escape(ln[last:m.start()]))
            if m.group(1):
                cls = 'g'
            elif m.group(2) or m.group(3) or m.group(4) in ('True', 'False', 'None'):
                cls = 'r'
            else:
                cls = 'b'
            out.append(f'<span class="{cls}">{_html.escape(m.group(0))}</span>')
            last = m.end()
        out.append(_html.escape(ln[last:]))
        return ''.join(out)

    lines = [hl_code(ln) or '&nbsp;' for ln in src.strip('\n').split('\n')]
    gut = ''.join(f'<div>{i}</div>' for i in range(1, len(lines) + 1))
    body = ''.join(f'<div style="white-space:pre">{l}</div>' for l in lines)
    gutter = (f'<div style="color:#666;text-align:right;padding:8px 8px;border-right:1px solid #DDD">{gut}</div>'
              if numbers else '')
    return (f'<div class="cb code" style="display:flex;border:1px solid #DDD;border-radius:4px;font-family:Consolas,Inc,monospace;'
            f'font-size:{size}px;line-height:1.5;background:#fff;min-width:0;overflow:hidden">' + gutter
            + f'<div style="padding:8px 12px">{body}</div></div>')


def cells(vals, fills, w=46, size=18):
    """A row of tensor cells. fills: one fill/line class string per value (or a single string)."""
    if isinstance(fills, str):
        fills = [fills] * len(vals)
    return '<div style="display:flex">' + ''.join(
        f'<span class="{f}" style="font-family:Inc;font-size:{size}px;width:{w}px;text-align:center;padding:4px 0;border:2px solid;margin:-1px 0 0 -2px">{v}</span>'
        for v, f in zip(vals, fills)) + '</div>'


# ---------------------------------------------------------------- Task 01-1
@plate('plate_01-1b_comparing_bags')
def _():
    letters = ['e', 'h', 'l', 'o', 'w', 'y']
    hello = [1, 1, 2, 1, 0, 0]
    yellow = [1, 0, 2, 1, 1, 1]
    head = ''.join(f'<th class="num" style="text-align:center">{c}</th>' for c in letters)
    r1 = ''.join(f'<td class="m" style="text-align:center">{v}</td>' for v in hello)
    r2 = ''.join(f'<td class="m" style="text-align:center">{v}</td>' for v in yellow)
    return title('Worked Example: Comparing Two Bags') + f'''
<p>A bag is just a table of counts, one entry per distinct item. Cosine similarity compares the two tables entry by entry.</p>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0;flex:1.15"><h3>Character bags: <span class="code b">hello</span> and <span class="code r">yellow</span></h3>
 <table class="t" style="font-size:18px;margin:4px 0 10px"><tr><th></th>{head}</tr>
 <tr><td class="code b" style="font-weight:bold">hello</td>{r1}</tr><tr><td class="code r" style="font-weight:bold">yellow</td>{r2}</tr></table>
 <div class="m" style="font-size:18px;line-height:1.8">
 <span class="b">a</span>·<span class="r">b</span> = 1·1 + 1·0 + 2·2 + 1·1 + 0·1 + 0·1 = 6<br>
 ‖<span class="b">a</span>‖ = √7, ‖<span class="r">b</span>‖ = √8<br>
 cos = 6 / √56 = <span class="g">0.802</span></div>
 <p style="font-size:16px;margin:8px 0 0" class="k">Missing items count as 0, so letters in only one word add nothing to the dot product.</p></div>
 <div class="card" style="margin:0"><h3><span class="code b">hello</span> and its disguise <span class="code r">uryyb</span></h3>
 <p style="font-size:17px;margin-bottom:8px">ROT13 renames every letter, so the two words share <b>no</b> letters:</p>
 <div class="m" style="font-size:18px;margin-bottom:12px">letter-by-letter cosine = <span class="r">0</span></div>
 <p style="font-size:17px;margin-bottom:8px">Now forget <b>which</b> letter each count belongs to and sort the counts, largest first:</p>
 <div class="m" style="font-size:18px;line-height:1.8"><span class="b">hello</span>: l h e o → [2, 1, 1, 1]<br><span class="r">uryyb</span>: y u r b → [2, 1, 1, 1]<br>sorted-profile cosine = <span class="g">1</span></div></div>
</div>
<div class="callout lg fg" style="margin-bottom:0">The <b>shape</b> of the counts survives the renaming even when the letters themselves don't. The same comparison, run on whole texts, is how we'll pick the note's cipher.</div>
'''


# ---------------------------------------------------------------- Task 02-2
@plate('plate_02-2b_from_pairs_to_batches')
def _():
    flow = chain([("<span class='code'>datasets['train']</span><small>900 (source, target) pairs</small>", 'fk lk'),
                  ('<span class="code">DataLoader</span><small>picks 64 pairs</small>', 'fb lb'),
                  ('<span class="code">dataset.collate</span><small>supplied</small>', 'fp lp'),
                  ('<span class="code">X, Y</span><small>(64, T) and (64, T, 41)</small>', 'fg lg')], size=16)
    pair = lambda s, t: f'<div class="box fk lk" style="min-width:0;padding:6px 10px;font-family:Inc;font-size:16px;font-weight:normal;white-space:nowrap">(<span class="r">{s}</span>, <span class="g">{t}</span>)</div>'
    unzip = f'''<div style="display:flex;align-items:center;gap:10px;justify-content:center">
 <div style="display:flex;flex-direction:column;gap:6px">{pair("uryyb", "hello")}{pair("69", "69")}</div>
 <span class="arr" style="font-size:22px">⇢</span>
 <div style="display:flex;flex-direction:column;gap:6px;font-family:Inc;font-size:16px;white-space:nowrap">
  <div class="box fr lr" style="min-width:0;padding:6px 10px;font-weight:normal">X_list = (<span class="r">uryyb</span>, <span class="r">69</span>)</div>
  <div class="box fg lg" style="min-width:0;padding:6px 10px;font-weight:normal">Y_list = (<span class="g">hello</span>, <span class="g">69</span>)</div></div></div>'''
    return title('From a List of Pairs to Batches') + f'''
<p>A <span class="code">DataLoader</span> serves a dataset in batches. It doesn't know how to stack phrases of different lengths, so the dataset gives it a <b class="def">collate function</b>:
it receives the list of pairs for one batch and returns the batch tensors.</p>
{flow}
<div class="row" style="margin:16px 0 12px">
 <div class="card" style="margin:0"><h3><span class="code">zip(*batch)</span> unzips the pairs</h3>
 <span style="font-size:16px">The <span class="code">*</span> passes each pair to <span class="code">zip</span> as a separate argument, so <span class="code">zip</span> groups the first items together and the second items together:</span>
 <div style="margin-top:10px">{unzip}</div>
 <p style="font-size:15px;margin:8px 0 0" class="k">shown as text; in the notebook each item is a tensor of token IDs</p></div>
 <div class="card" style="margin:0;flex:.85"><h3>Pad, then one-hot the targets</h3>
 <span class="code">pad_sequence</span> pads each group into one <span class="code">(B, T)</span> tensor of IDs. The targets then go through <span class="code">F.one_hot</span>: each ID becomes a vector of 41 zeros with a single 1, so <span class="code">Y</span> is <span class="code">(B, T, 41)</span>, ready for next week's loss.</div>
</div>
<div class="row">
 <div class="callout lb fb" style="margin:0"><b><span class="m">T</span> changes from batch to batch.</b> Each batch is padded only to its own longest phrase, and the training loader shuffles the phrases every epoch.</div>
 <div class="callout lk fk" style="margin:0"><b>The last batch is smaller.</b> 900 pairs in batches of 64 is 14 full batches and one batch of 4. Nothing downstream should assume <span class="m">B = 64</span>.</div>
</div>
'''


# ---------------------------------------------------------------- Task 03-1
@plate('plate_03-1a_why_not_ids')
def _():
    ticks = [(5, 'a'), (6, 'b'), (30, 'z'), (31, '0')]
    line = ''.join(f'<line x1="{30+i*22}" y1="54" x2="{30+i*22}" y2="62" stroke="#999"/>' for i in range(41))
    marks = ''.join(f'<circle cx="{30+i*22}" cy="58" r="7" fill="#F4CCCC" stroke="#E06666" stroke-width="2"/>'
                    f'<text x="{30+i*22}" y="38" text-anchor="middle" font-family="Inc" font-size="17" font-weight="bold">{c}</text>'
                    f'<text x="{30+i*22}" y="86" text-anchor="middle" font-family="Inc" font-size="15" fill="#666">{i}</text>' for i, c in ticks)
    svg = f'<svg width="940" height="96"><line x1="30" y1="58" x2="910" y2="58" stroke="#999" stroke-width="2"/>{line}{marks}</svg>'
    E = mat('b', 2, [0, 0, 1, 2, 3, 4], hl=(4, 5))
    oh = mat('r', 3, [0, 0, 1])
    return title('Why Not Feed the IDs Directly?') + f'''
<p>A linear layer multiplies its inputs by weights. Give it a raw token ID and it treats the ID as an <b>amount</b>:</p>
<div style="text-align:center">{svg}</div>
<div class="callout lr fr" style="margin-top:4px">The ID scale invents relationships that aren't there: <span class="code">b</span> looks almost the same as <span class="code">a</span>, <span class="code">z</span> sits right next to the digit <span class="code">0</span>, and <span class="code">z</span> would push the weights <b>six times</b> as hard as <span class="code">a</span>.</div>
<div class="row" style="margin-bottom:12px">
 <div class="card" style="margin:0"><h3>The fix: one-hot vectors</h3>Give every token its own input, set to 1 for that token and 0 for the rest.
 Now every pair of tokens is equally different. The cost: <span class="m">41</span> inputs per character, <span class="m">40</span> of them zero.</div>
 <div class="card" style="margin:0;flex:1.25"><h3>Multiplying a one-hot vector by a matrix</h3>
 <div class="eqrow" style="justify-content:center;gap:10px"><span class="note">onehot(2) =</span> {oh} @ {E} = <span class="r">[3, 4]</span></div>
 <p style="font-size:15px;margin:4px 0 0;text-align:center" class="k">the matrix here is <span class="m b">W</span><sup>T</sup>: one row per token</p>
 <p style="font-size:16px;margin:8px 0 0">Every row is multiplied by 0 except row 2: the product just <b>selects row 2</b>.</p></div>
</div>
<div class="callout lg fg" style="margin-bottom:0"><b>An embedding is a linear layer on one-hot inputs.</b> That is exactly how your <span class="code">EmbeddingLayer</span> works: one-hot vectors times its weights. PyTorch's <span class="code">nn.Embedding</span> skips the multiplication and looks the row up directly. Either way the weights are learned, so tokens can end up as similar or as different as the task needs.</div>
'''


# ---------------------------------------------------------------- Task 03-2
@plate('plate_03-2b_why_bounded')
def _():
    relu = [0.5, 1.25, 2.375, 4.062, 6.594, 10.391, 16.086, 24.629]
    tanh = [0.462, 0.832, 0.941, 0.957, 0.959, 0.959, 0.959, 0.959]
    head = ''.join(f'<th class="num">{t}</th>' for t in range(1, 9))
    rr = ''.join(f'<td class="m num r" style="font-size:16px">{v}</td>' for v in relu)
    tr = ''.join(f'<td class="m num" style="font-size:16px;color:#B8860B;font-weight:bold">{v}</td>' for v in tanh)
    return title('Why a Bounded Activation?') + f'''
<div class="card" style="margin-bottom:0"><h3>Feeding a value back into itself</h3>
<p style="font-size:17px;margin-bottom:6px">An RNN applies the same weights to its own output at every step. A one-number version, <span class="m">h<sub>t</sub> = f(1.5 h<sub>t−1</sub> + 0.5)</span> with <span class="m">h<sub>0</sub> = 0</span>:</p>
<table class="t" style="font-size:16px;margin:0 0 6px"><tr><th>step t</th>{head}</tr>
<tr><td><b>f = ReLU</b></td>{rr}</tr><tr><td><b>f = tanh</b></td>{tr}</tr></table>
<p style="font-size:16px;margin:0">With ReLU the state grows by about 1.5× per step; over the 263 characters of the note it would explode. tanh can never leave <span class="m">(−1, 1)</span>, so it settles.
Sigmoid is bounded too, but its outputs are all positive; tanh is <b>centered on zero</b>, so a hidden unit can push either way.</p></div>
'''


# ---------------------------------------------------------------- Task 04-2
@plate('plate_04-2b_inside_forward')
def _():
    rows = [('<span class="code">X</span> into the network', 'token IDs', '(2, 6)'),
            ('inside <span class="code">EmbeddingLayer</span>', 'one-hot vectors, <span class="code">self.X</span>', '(2, 6, 41)'),
            ('out of <span class="code">EmbeddingLayer</span>', 'embeddings', '(2, 6, 32)'),
            ('<span class="code">x_at_timestep</span>', 'one position of both phrases', '(2, 32)'),
            ('<span class="code">h_prev_at_timestep</span>', 'the previous hidden state', '(2, 64)'),
            ('<span class="code">z_at_timestep</span>, <span class="code">h_at_timestep</span>', 'before and after tanh', '(2, 64)'),
            ('<span class="code">self.hidden_states</span>', 'the zero start plus one <span class="code">Grad_Info</span> per step', '7 entries'),
            ('out of <span class="code">RecurrentBlock</span>', 'every hidden state, stacked on <span class="code">dim=1</span>', '(2, 6, 64)'),
            ('out of the output <span class="code">LinearLayer</span>', 'logits', '(2, 6, 41)')]
    body = ''.join(f'<tr><td>{a}</td><td>{b}</td><td class="code">{c}</td></tr>' for a, b, c in rows)
    mistakes = [('<span class="code">X[timestep]</span> instead of <span class="code">X[:, timestep, :]</span>', 'selects phrase <span class="m">t</span>, not position <span class="m">t</span>'),
                ('Using <span class="code">self.hidden_states[0]</span> as the previous state at every step', 'every position forgets what came before'),
                ('<span class="code">layer.forward(X)</span> without <span class="code">X =</span>', 'the network returns its input unchanged'),
                ('Forgetting to append a <span class="code">Grad_Info</span>', 'the next step reuses the zero start, and HW09 has nothing to backpropagate through')]
    mrows = ''.join(f'<tr><td style="width:52%">{a}</td><td>{b}</td></tr>' for a, b in mistakes)
    return title('Inside the Forward Pass') + f'''
<p>Trace the toy batch from Task 02 (<span class="m">B = 2</span> phrases, <span class="m">T = 6</span> positions) through <span class="code">RecurrentNeuralNetwork.forward</span>. If your shapes match at each stage, the network is wired correctly.</p>
<table class="t" style="font-size:16px"><tr><th>Stage</th><th>Holds</th><th>Shape</th></tr>{body}</table>
<table class="t" style="font-size:16px;margin-bottom:0"><tr><th>Common mistake</th><th>What goes wrong</th></tr>{mrows}</table>
'''


# ---------------------------------------------------------------- Task 05-1
@plate('plate_05-1b_scratch_and_library')
def _():
    stages = [('<span class="code">EmbeddingLayer.forward</span>: one-hot, then <span class="code">@ W.T</span>', '<span class="code">self.embed</span>: looks the row up directly', '(B, T, 32)'),
              ('<span class="code">RecurrentBlock.forward</span>: your loop over timesteps', '<span class="code">self.rnn</span>: the whole loop in one call', '(B, T, 64)'),
              ('output <span class="code">LinearLayer.forward</span>', '<span class="code">self.fc</span>: applied to every position at once', '(B, T, 41)')]
    body = ''.join(f'<tr><td>{a}</td><td>{b}</td><td class="code">{c}</td></tr>' for a, b, c in stages)
    return title('The Scratch Model and the Library Model') + f'''
<table class="t" style="font-size:16px"><tr><th>RecurrentNeuralNetwork (Task 04)</th><th>CharRNN</th><th>Result</th></tr>{body}</table>
<div class="row" style="margin-bottom:10px">
 <div class="card" style="margin:0"><h3><span class="code">nn.RNN</span> returns two things</h3>
 The hidden state at <b>every</b> position, <span class="code">(B, T, 64)</span>, and the final state alone, <span class="code">(1, B, 64)</span>. <span class="code">CharRNN</span> only needs the first; <span class="code">_</span> is the usual name for a value we ignore.
 Unlike your <span class="code">hidden_states</span> list, it does <b>not</b> include <span class="m">h<sub>0</sub></span>.</div>
 <div class="card" style="margin:0"><h3>Only the embedding needs a transpose</h3>
 PyTorch stores <span class="code">nn.RNN</span> and <span class="code">nn.Linear</span> weights the way we do, "outputs × inputs". <span class="code">nn.Embedding</span> keeps one <b>row</b> per token, so the supplied check copies <span class="code">EmbeddingLayer.W.T</span> into it.
 <span class="code">copy_</span> requires matching shapes, so a wrong pairing fails loudly.</div>
</div>
<div class="callout lb fb" style="margin-bottom:0"><b>Why does PyTorch keep two biases?</b> For this simple RNN they're redundant: only their sum matters. They're there for its gated relative, the <b>GRU</b>,
which multiplies the hidden-side part by a gate (<span class="m">r ⊙ (<span class="y">h</span> <span class="b">W<sub>hn</sub></span><sup>T</sup> + <span class="k2">b<sub>hn</sub></span>)</span>), and for compatibility with NVIDIA's cuDNN library. All of PyTorch's recurrent layers share that layout.</div>
'''


# ---------------------------------------------------------------- Task 05-2
SPEC = {'### Task 01-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_01-1a_cosine_similarity',
                                            'Cosine similarity compares the directions of two count vectors and ignores their lengths: cos theta = a dot b '
                                            'over the product of their lengths. a = [1, 2] and b = [2, 4] point the same way: dot product 10, lengths root 5 '
                                            'and root 20, cosine 1. a and c = [3, 1] are 45 degrees apart: 5 over root 50 = 0.707. [1, 0] and [0, 1] share '
                                            'nothing: cosine 0. hello and hellohello have counts [1, 1, 2, 1] and [2, 2, 4, 2], about 2.65 apart but cosine '
                                            'exactly 1. For counts, the cosine is between 0 and 1.'),
                                           ('plate_01-1b_comparing_bags',
                                            'Worked example: character bags of hello and yellow over e, h, l, o, w, y are [1,1,2,1,0,0] and [1,0,2,1,1,1]; dot '
                                            'product 6, norms root 7 and root 8, cosine 6 over root 56 = 0.802. hello and its ROT13 disguise uryyb share no '
                                            'letters, so their letter-by-letter cosine is 0, but their sorted counts are both [2,1,1,1], so the sorted-profile '
                                            'cosine is 1.'),
                                           ('plate_01-1c_five_candidate_ciphers',
                                            'Five candidate ciphers from the pycipher library, each with a fixed key. Caesar(3): hello becomes KHOOR. Caesar(13), ROT13, its own inverse: URYYB. Vigenere with key clue: the shift cycles through the key, JPFPQ, so the two ls become different letters and counts are blurred. Affine(5, 8): x becomes (5x + 8) mod 26, RCLLA. SimpleSubstitution: a fixed scrambled alphabet, ITSSG. All but Vigenere rename letters consistently. encipher returns capitals and drops spaces, digits and punctuation. The test: encode an English sample with each candidate, compare its word and character bags with the note, and score BoW + BoC. The keys are fixed, so a low score rules out that key, not the whole family.')])],
 '### Task 02-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_02-1_characters_become_ids',
                                            'Vocabulary of V = 41 tokens: 0 PAD, 1 EOS, 2 UNK, 3 SOS (unused), 4 space, 5 to 30 letters a to z, 31 to 40 '
                                            'digits 0 to 9. Token IDs are category labels. encode hello 69 gives [12, 9, 16, 16, 19, 4, 37, 40, 1]. decode '
                                            'stops at the first EOS, skips PAD, and shows other special tokens as ?.'),
                                           ('plate_02-2a_padding',
                                            'Padding: append PAD (ID 0) so a batch of B phrases becomes a (B, T) tensor. Sources [[25, 22, 29, 29, 6, 1], [37, '
                                            '40, 1, 0, 0, 0]] and targets [[12, 9, 16, 16, 19, 1], [37, 40, 1, 0, 0, 0]]. ROT13 keeps lengths equal, so '
                                            'positions align. Losses and accuracies ignore PAD positions.'),
                                           ('plate_02-2b_from_pairs_to_batches',
                                            'From pairs to batches: the training Rot13Dataset holds 900 (source, target) pairs; the DataLoader picks 64 and '
                                            'passes them to the supplied collate, which returns X of shape (64, T) and one-hot targets Y of shape (64, T, 41). '
                                            'zip(*batch) unzips the pairs; pad_sequence pads each group; F.one_hot turns the target IDs into one-hot vectors. '
                                            'T changes from batch to batch, and the last batch has only 4 pairs.')])],
 '### Task 03-2: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_03-1a_why_not_ids',
                                            'Why not feed the IDs directly: a linear layer treats an ID as an amount, so b (6) looks almost the same as a (5), '
                                            'z (30) sits next to the digit 0 (31), and z pushes the weights six times as hard as a. One-hot vectors make every '
                                            'pair of tokens equally different but use 41 inputs, 40 of them zero. A one-hot vector times a matrix just selects '
                                            'one row. An embedding is a linear layer on one-hot inputs.'),
                                           ('plate_03-1b_embeddings',
                                            'An embedding layer is a LinearLayer with no bias applied to one-hot vectors. Its weight W has shape (E, V), one '
                                            'column per token, and onehot(i) times W transpose is column i of W. Example W = [[0, 1, 3], [0, 2, 4]]: tokens '
                                            '[2, 1, 2] give [[3, 4], [1, 2], [3, 4]]. (B, T) IDs become (B, T, V) one-hot vectors and then (B, T, E) '
                                            'embeddings; E = 32.')])],
 '### Task 03-3: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_03-2a_tanh',
                                            'tanh(a) = (e^a - e^-a) / (e^a + e^-a) maps any number into (-1, 1) and is centered on zero, unlike sigmoid. Slope '
                                            '1 - tanh(a)^2. Worked example: tanh(1) = (2.718 - 0.368) / (2.718 + 0.368) = 2.350 / 3.086 = 0.7616; tanh(-1) = '
                                            '-0.7616; tanh(3) = 0.9951.'),
                                           ('plate_03-2b_why_bounded',
                                            'Why a bounded activation: the one-number recurrence h_t = f(1.5 h_(t-1) + 0.5) from h_0 = 0 grows 0.5, 1.25, '
                                            '2.375, 4.06, ..., 24.6 after 8 steps with ReLU but settles at 0.959 with tanh. tanh is centered on zero, unlike '
                                            'sigmoid.')])],
 '### Task 04-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_04-1_rnn_step',
                                            'One recurrent step: h_t = tanh(x_t W^T + h_(t-1) U^T + b). Shapes: x_t (B, 32), h (B, 64), W (64, 32), U (64, '
                                            '64), b (64,). h_0 = 0. Example: x = [1, 0], h_prev = [0.5, -0.5], W = I, U = 0.5 I, b = [0, 0.1] gives z = [1.25, '
                                            '-0.15] and h_t = [0.8483, -0.1489].'),
                                           ('plate_04-2a_unrolling',
                                            'Unrolling: the same step at every position with shared weights W and U. The output layer turns each hidden state '
                                            'into 41 logits, so (B, T) token IDs become (B, T, 41) logits. At each timestep X[:, timestep, :] has shape (B, '
                                            '32). hidden_states caches a Grad_Info(x, h, h_prev) for every step plus the zero start: T + 1 entries, used by '
                                            'backpropagation in HW09.')])],
 '### Task 04-2: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_04-2b_inside_forward',
                                            'Inside the forward pass for the toy batch (B = 2, T = 6): token IDs (2, 6); one-hot (2, 6, 41); embeddings (2, 6, '
                                            '32); at each timestep x (2, 32), h_prev (2, 64), z and h (2, 64); hidden_states holds 7 entries; outputs stack to '
                                            '(2, 6, 64); the output layer gives logits (2, 6, 41). Common mistakes: X[timestep] instead of X[:, timestep, :]; '
                                            'using the zero start at every step; calling layer.forward(X) without X = ; stacking with dim=0.'),
                                           ('plate_04-2c_parameters',
                                            'Parameters: embedding W 32 x 41 = 1,312, recurrent W 64 x 32 = 2,048, U 64 x 64 = 4,096, b 64, output W 41 x 64 = '
                                            '2,624, output b 41: 10,185 in all, independent of sequence length. The embedding layer also inherits an unused '
                                            'bias from LinearLayer.')])],
 '### Task 05-1: Description (0 pts.)': [(['(already applied: plates were wired into the notebooks directly)'],
                                          [('plate_05-1a_library_version',
                                            'PyTorch version: nn.Embedding(V, 32) for the embedding layer, whose weight is our W transposed; '
                                            'nn.RNN(input_size=32, hidden_size=64, batch_first=True) for the recurrent block, with weight_ih_l0 = W, '
                                            'weight_hh_l0 = U, and bias_ih_l0 + bias_hh_l0 = b; nn.Linear(64, V) for the output layer.'),
                                           ('plate_05-1b_scratch_and_library',
                                            'Scratch and library side by side: the embedding, the recurrent loop, and the output layer each map to one PyTorch '
                                            'module. nn.RNN returns all hidden states (B, T, 64), without h_0, and the final state (1, B, 64). Only the '
                                            'embedding needs a transpose when copying. PyTorch keeps two recurrent biases for its gated relative, the GRU, and '
                                            'for cuDNN compatibility.')])]}
EXTRA = {}

if __name__ == '__main__':
    main('08', PLATES, SPEC, EXTRA)
