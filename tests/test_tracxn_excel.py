from capstone.tracxn_excel import _ceo, _date_of, _num


def test_tracxn_values():
    assert _num("31.59% (Mar 31, 2026)") == 31.59
    assert _num("4.49x") == 4.49
    assert _num("3,28,594 (Mar 31, 2026)") == 328594
    assert _num("18,29,72,00,00,000 (Mar 31, 2026)", 1e7) == 182972
    assert _num("21,47,48,36,47,408 (Oct 8, 2026)") is None  # Tracxn overflow value
    assert _date_of("2,94,74,00,00,000 (Mar 31, 2026)") == "2026-03-31"
    assert _ceo("A B; Co-Founder & Chairman; x\nSalil Parekh; CEO & Managing Director; y") == "Salil Parekh"
