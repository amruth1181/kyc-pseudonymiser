import pytest

from kycpseudo.ff1 import FF1

KEY = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.mark.parametrize("radix,alphabet,tweak,pt,ct", [
    # NIST SP 800-38G sample vectors, AES-128
    (10, None, "", "0123456789", "2433477484"),
    (10, None, "39383736353433323130", "0123456789", "6124200773"),
    (36, B36, "3737373770717273373737", "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
])
def test_nist_vectors(radix, alphabet, tweak, pt, ct):
    f = FF1(KEY, radix, alphabet)
    t = bytes.fromhex(tweak)
    assert f.encrypt(pt, t) == ct
    assert f.decrypt(ct, t) == pt


def test_format_is_preserved():
    f = FF1(KEY, 10)
    c = f.encrypt("3613412524")
    assert len(c) == 10 and c.isdigit() and c != "3613412524"


def test_small_domain_is_refused():
    with pytest.raises(ValueError):
        FF1(KEY, 10).encrypt("1234")
