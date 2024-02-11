def answer_to_life_universe_everything():
    return 42

def test_answer():
    assert 42 == answer_to_life_universe_everything()

def test_answer_fail():
    assert 47 == answer_to_life_universe_everything()