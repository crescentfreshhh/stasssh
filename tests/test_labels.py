from peaks.labels import Label, LabelStore


def test_add_and_counts(tmp_path):
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 10.0, 1, "apex", scene_id="1")
    store.add("k1", 20.0, 0, "apex")
    store.add("k2", 5.0, 1, "apex")
    assert store.counts("apex") == (2, 1)  # 2 pos, 1 neg
    assert len(store) == 3


def test_upsert_overwrites_same_frame(tmp_path):
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 10.0, 1, "apex")
    store.add("k1", 10.004, 0, "apex")  # within rounding -> same frame id
    assert len(store) == 1
    assert store.counts("apex") == (0, 1)


def test_profiles_are_isolated(tmp_path):
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 10.0, 1, "apex")
    store.add("k1", 10.0, 0, "apex:heels")  # same frame, different profile
    assert len(store) == 2
    assert store.counts("apex") == (1, 0)
    assert store.counts("apex:heels") == (0, 1)
    assert store.profiles() == ["apex", "apex:heels"]


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "labels.json"
    store = LabelStore(path)
    store.add("k1", 10.0, 1, "apex", scene_id="42")
    store.save()

    reloaded = LabelStore(path)
    assert len(reloaded) == 1
    lab = reloaded.for_profile("apex")[0]
    assert isinstance(lab, Label)
    assert lab.key == "k1" and lab.label == 1 and lab.scene_id == "42"
