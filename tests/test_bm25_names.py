from kycpseudo.bm25 import BM25, char_ngrams
from kycpseudo.names import format_like, name_similarity, normalize


def test_ngrams_have_boundaries():
    assert char_ngrams("ANA") == ["#AN", "ANA", "NA#"]


def test_bm25_finds_ocr_variant():
    docs = ["AMRUTH GANTA", "ANKIT ARORA", "ANNA MARIA ERIKSSON", "AMIT GUPTA"]
    top = BM25(docs).top_k("AMRUTB CANTA", k=1)
    assert top[0][0] == 0


def test_bm25_rare_surname_dominates():
    docs = ["ANNA SMITH", "ANNA JONES", "ANNA LINDQVIST", "MARIA LINDQVIST"]
    ranked = [i for i, _ in BM25(docs).top_k("ERIK LINDQVIST", k=4)]
    assert set(ranked[:2]) == {2, 3}


def test_bm25_explain_lists_matching_grams():
    idx = BM25(["OSKAR MALMBERG", "OLOF MALMBERG"])
    assert any(g == "MAL" for g, _ in idx.explain("MALMBERG", 0))


def test_similarity_initials_flagged():
    s, by_initial = name_similarity("O. MALMBERG", "Oskar Malmberg")
    assert s > 0.95 and by_initial


def test_similarity_needs_a_full_token():
    s, _ = name_similarity("O. M.", "Oskar Malmberg")
    assert s == 0.0


def test_similarity_order_and_typo():
    s, by_initial = name_similarity("DUBOIS, Etienne Thomas", "Etienne Thomas Dueois")
    assert s > 0.92 and not by_initial


def test_format_like_keeps_pattern():
    g, s, ng, ns = ["ANNA", "MARIA"], "ERIKSSON", ["EVA", "KARIN"], "LINDQVIST"
    assert format_like("A. M. ERIKSSON", g, s, ng, ns) == "E. K. LINDQVIST"
    assert format_like("ERIKSSON, Anna Maria", g, s, ng, ns) == "LINDQVIST, Eva Karin"
    assert format_like("ERIKSSON<<ANNA<MARIA", g, s, ng, ns) == "LINDQVIST<<EVA<KARIN"


def test_normalize():
    assert normalize("Éric  d'Ávila<<JR") == "ERIC D AVILA JR"
