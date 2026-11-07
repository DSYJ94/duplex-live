from data.venus.retention_orthography import wer, words


def test_only_declared_same_word_forms_and_no_phonetic_or_key_equivalence():
    for a,b in [('a paper clip','a paperclip'),('computer program','computer programme'),
                ('optimizer state','optimiser state'),('nondeterministic','non-deterministic')]:
        assert wer(a,b)==0
    for a,b in [('room sign','rune sign'),('lookup bookcase','lookup bokeh'),('six','seven'),
                ('the marker','a marker'),('warm day','worn day')]:
        assert wer(a,b)>0
    assert words('Non deterministic computer programme')==['nondeterministic','computer','program']
