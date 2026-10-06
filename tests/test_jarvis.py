import torch

from jarvis import checkpoint
from jarvis.data import GPTDataset
from jarvis.generate import generate, generate_text
from jarvis.model import GPTConfig, GPTModel, MultiHeadAttention
from jarvis.tokenizer import BPETokenizer

TEXT = "Hola, soy Jarvis. Jarvis aprende español: ¡qué alegría! 123 niños y niñas. " * 20


def test_tokenizer_roundtrip_and_compression():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8"))
    sample = "Texto nuevo con acentos: canción, pingüino.<|endoftext|>Fin"
    assert tok.decode(tok.encode(sample)) == sample
    assert tok.eot_id in tok.encode(sample)


def test_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert loaded.vocab_size == tok.vocab_size


def test_dataset_targets_are_shifted_inputs():
    ds = GPTDataset(list(range(50)), max_length=8, stride=4)
    x, y = ds[1]
    assert x.tolist() == list(range(4, 12))
    assert y.tolist() == list(range(5, 13))


def test_attention_is_causal():
    torch.manual_seed(0)
    att = MultiHeadAttention(16, 16, context_length=8, dropout=0.0, num_heads=4).eval()
    x = torch.randn(1, 6, 16)
    out = att(x)
    x2 = x.clone()
    x2[:, -1] += 10  # cambiar el último token no debe afectar a los anteriores
    assert torch.allclose(att(x2)[:, :-1], out[:, :-1], atol=1e-6)


def test_model_shapes_and_generation(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    cfg = GPTConfig(vocab_size=tok.vocab_size, context_length=16, emb_dim=32, n_heads=4, n_layers=2)
    model = GPTModel(cfg).eval()
    logits = model(torch.zeros(2, 10, dtype=torch.long))
    assert logits.shape == (2, 10, tok.vocab_size)

    out = generate(model, torch.zeros(1, 20, dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 25)  # funciona aunque la entrada supere el contexto
    assert isinstance(generate_text(model, tok, "Hola", max_new_tokens=5), str)

    checkpoint.save(tmp_path, model, tok)
    loaded, _ = checkpoint.load(tmp_path)
    x = torch.randint(0, tok.vocab_size, (1, 8))
    assert torch.allclose(loaded(x), model(x))


def test_model_can_overfit_tiny_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, context_length=8, emb_dim=32, n_heads=2, n_layers=2, drop_rate=0.0)
    model = GPTModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 50, (2, 8))
    y = torch.randint(0, 50, (2, 8))
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_instruction_collate_masks_prompt_and_padding():
    from jarvis.instruct import IGNORE_INDEX, make_collate_fn
    collate = make_collate_fn(pad_id=0, max_length=32)
    x, y = collate([([1, 2, 3], [4, 5, 9]), ([1, 2], [6, 9])])
    assert x.shape == y.shape == (2, 5)
    # primer ejemplo: solo se aprende la respuesta (4, 5, 9)
    assert y[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 4, 5, 9]
    # segundo ejemplo: respuesta (6, 9) y luego relleno ignorado
    assert y[1].tolist() == [IGNORE_INDEX, 6, 9, IGNORE_INDEX, IGNORE_INDEX]


def test_instruction_collate_keeps_answer_when_prompt_is_too_long():
    from jarvis.instruct import IGNORE_INDEX, make_collate_fn
    collate = make_collate_fn(pad_id=0, max_length=8)
    x, y = collate([(list(range(100, 120)), [1, 2, 3])])
    assert x.shape == (1, 8)
    assert x[0].tolist() == [114, 115, 116, 117, 118, 119, 1, 2]  # final de la instrucción
    assert y[0].tolist()[-3:] == [1, 2, 3]
    assert (y[0] != IGNORE_INDEX).sum() == 3
