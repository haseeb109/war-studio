from warstudio.steps import specs, rank


def test_validate_builds_tiered_queries():
    reply = [{"id": 1, "visual": "Su-34 jet taking off, side view", "entity": "Su-34",
              "exact_queries": ["Su-34 takeoff", "Su-34 fighter"], "substitute": "Russian twin-seat fighter jet",
              "substitute_queries": ["Russian fighter jet takeoff"], "category_query": "fighter jet",
              "scene": "Russian Jets Scramble", "footage_class": "real", "avoid": ["airshow"], "label": "Su-34", "sfx": []}]
    out, errors = specs.validate(reply, {1})
    assert not errors
    s = out[1]
    assert s["queries"] == ["Su-34 takeoff", "Su-34 fighter", "Russian fighter jet takeoff", "fighter jet"]
    assert s["tiers"] == ["exact", "exact", "substitute", "category"]
    assert s["scene"] == "russian jets scramble" and s["visual"].startswith("Su-34")


def test_validate_requires_category_and_visual():
    out, errors = specs.validate([{"id": 1, "exact_queries": ["x"], "footage_class": "real"}], {1})
    assert errors and 1 not in out


def test_generic_beat_has_no_exact_tier():
    out, _ = specs.validate([{"id": 2, "visual": "radar operators at consoles", "exact_queries": [],
                              "category_query": "radar screen", "scene": "s", "footage_class": "control_room"}], {2})
    assert out[2]["tiers"] == ["category"]


def test_fallback_spec_always_has_category():
    f = specs.fallback_spec({"id": 3, "text": "The F-35 lifted off at dawn."})
    assert f["tiers"][-1] == "category" and f["visual"]


def test_adjusted_score_ladder():
    base = {"clip_score_raw": 0.30, "duration": 20}
    assert rank.adjusted_score(dict(base, tier="exact")) > rank.adjusted_score(dict(base, tier="substitute")) \
        > rank.adjusted_score(dict(base, tier="category"))
    reel = dict(base, tier="category", duration=400)
    assert rank.adjusted_score(reel) < rank.adjusted_score(dict(base, tier="category"))
    assert rank.adjusted_score(dict(base, tier="exact", duration=400)) == rank.adjusted_score(dict(base, tier="exact"))
    assert rank.adjusted_score(dict(base, tier="category", scene_reuse=True)) > rank.adjusted_score(dict(base, tier="category"))
