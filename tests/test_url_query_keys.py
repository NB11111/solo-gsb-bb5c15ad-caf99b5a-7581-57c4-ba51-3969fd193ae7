"""Tests for the URL query-key ambiguity audit endpoint.

Covers: POST /api/v1/urls/query-keys splitting the raw query on the
undecoded '&' and first '=' (including empty keys and parameters
without values), original-URL codepoint spans/positions, the single
percent/UTF-8 decode pass ('+' as space, '%XX' hex triplets, no
double decoding, multibyte and astral UTF-8), NFKC collision grouping
(ASCII/fullwidth/percent encodings, precomposed vs combining forms,
raw duplicates) with stable ordering and member metadata, parameter
values never decoded or compared, fragment truncation and the first
'?' delimiting the query, 422 with original offsets on malformed
percent encoding or invalid UTF-8 without any partial report, the
read-only response shape, and Pydantic 422 on a missing/empty url.
"""

import pytest

from app.url_query_keys import QueryKeyParseError, audit_query_keys


# ------------------------------------------------------------- splitting --

def test_split_on_undecoded_ampersand_and_first_equals():
    url = "https://h.test/path?a=1&b=2&c"
    report = audit_query_keys(url)
    assert report.parameter_count == 3
    assert report.collisions == []


def test_second_equals_belongs_to_the_value():
    # only the first '=' separates key from value; '==' never split the key
    report = audit_query_keys("http://h/?a=b=c&a=d=e")
    (group,) = report.collisions
    assert group.normalised == "a"
    assert [m.raw_key for m in group.members] == ["a", "a"]
    assert [m.index for m in group.members] == [1, 2]


def test_encoded_ampersand_and_equals_do_not_split():
    report = audit_query_keys("http://h/p?a%26b=1&a%3Db=2")
    # 'a&b' and 'a=b' are distinct keys, no split on encoded delimiters
    assert report.parameter_count == 2
    assert report.collisions == []


def test_empty_keys_from_empty_segments_are_parameters():
    report = audit_query_keys("http://h/?=1&&=2")
    assert report.parameter_count == 3
    (group,) = report.collisions
    assert group.normalised == ""
    assert [m.index for m in group.members] == [1, 2, 3]


def test_trailing_ampersand_adds_empty_parameter():
    report = audit_query_keys("http://h/?a=1&")
    assert report.parameter_count == 2
    # the trailing parameter is an empty key, distinct from 'a'
    assert report.collisions == []


def test_no_query_component():
    report = audit_query_keys("https://h.test/path")
    assert report.parameter_count == 0
    assert report.collisions == []


def test_empty_query_after_question_mark_has_one_empty_key():
    report = audit_query_keys("http://h/?")
    # str.split on the empty query yields one empty parameter
    assert report.parameter_count == 1
    assert report.collisions == []  # a single key cannot collide with itself


def test_fragment_terminates_the_query():
    report = audit_query_keys("http://h/?a=1#frag?b=2&a=3")
    assert report.parameter_count == 1
    assert report.collisions == []


def test_only_the_first_question_mark_starts_the_query():
    report = audit_query_keys("http://h/p?a?b=1&a?b=2")
    (group,) = report.collisions
    assert [m.raw_key for m in group.members] == ["a?b", "a?b"]


def test_key_offsets_span_only_the_key_in_the_original_url():
    url = "https://e.test/p?name=a&name=b#f"
    report = audit_query_keys(url)
    members = report.collisions[0].members
    assert [(m.start, m.end) for m in members] == [(17, 21), (24, 28)]
    assert url[members[0].start:members[0].end] == "name"
    assert url[members[1].start:members[1].end] == "name"


def test_key_without_value_offset_excludes_ampersand():
    report = audit_query_keys("http://h/?x&x")
    first, second = report.collisions[0].members
    assert (first.start, first.end) == (10, 11)
    assert (second.start, second.end) == (12, 13)


# -------------------------------------------------------------- decoding --

def test_plus_decodes_to_space():
    report = audit_query_keys("http://h/?a+b=1&a%20b=2")
    (group,) = report.collisions
    assert group.normalised == "a b"
    assert [m.decoded_key for m in group.members] == ["a b", "a b"]
    assert group.members[0].raw_key == "a+b"


def test_percent_hex_triplets_decode_case_insensitively():
    report = audit_query_keys("http://h/?%6e=1&%6E=2&n=3")
    (group,) = report.collisions
    assert group.normalised == "n"
    assert len(group.members) == 3


def test_decoding_happens_exactly_once():
    # %2541 decodes to "%41" in one pass; it must NOT decode again to "A"
    report = audit_query_keys("http://h/?%2541=1&A=2")
    assert report.collisions == []
    report = audit_query_keys("http://h/?%2541=1&%2541=2")
    (group,) = report.collisions
    assert group.normalised == "%41"


def test_multibyte_utf8_key_decodes_and_collides():
    report = audit_query_keys("http://h/?%C3%A9=1&%c3%a9=2")
    (group,) = report.collisions
    assert group.normalised == "é"
    assert all(m.decoded_key == "é" for m in group.members)


def test_combining_sequence_and_precomposed_form_unify_under_nfkc():
    # U+00E9 (precomposed) vs U+0065 U+0301 (decomposed): different raw
    # encodings and different decoded spellings, but NFKC unifies them.
    precomposed = "é"
    decomposed = "e" + "́"
    assert precomposed != decomposed
    report = audit_query_keys("http://h/?%C3%A9=1&e%CC%81=2")
    (group,) = report.collisions
    assert group.normalised == precomposed
    assert group.members[0].decoded_key == precomposed
    assert group.members[1].decoded_key == decomposed


def test_astral_plane_utf8_uses_codepoint_offsets():
    report = audit_query_keys("http://h/?\U0001F600=1&%f0%9f%98%80=2")
    (group,) = report.collisions
    literal, encoded = group.members
    assert literal.decoded_key == encoded.decoded_key == "\U0001F600"
    # the literal astral character is one codepoint; the %XX run spans 12
    assert (literal.start, literal.end) == (10, 11)
    assert (encoded.start, encoded.end) == (14, 26)


def test_literal_non_ascii_characters_are_not_percent_decoded():
    report = audit_query_keys("http://h/?é=1&%C3%A9=2")
    (group,) = report.collisions
    assert group.normalised == "é"
    assert group.members[0].raw_key == "é"


def test_percent_decoded_byte_run_must_be_fully_valid_utf8():
    # %28 ('(') interrupts the lead byte's continuation sequence; the
    # maximal run %E2%28%A1 is invalid UTF-8 as a whole
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys("http://h/?%E2%28%A1=x")
    assert excinfo.value.error == "invalid_utf8"


def test_percent_runs_are_decoded_as_one_utf8_unit():
    # %28%A1 is one maximal %XX run: decoded bytes 0x28 0xA1, invalid
    # UTF-8 as a whole, and the error spans the whole run
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys("http://h/?%28%A1=x")
    err = excinfo.value
    assert err.error == "invalid_utf8"
    assert (err.start, err.end) == (10, 16)


def test_literal_character_starts_a_fresh_byte_run():
    # a literal '(' ends the %28 run; the following %A1 is then a fresh,
    # self-invalid continuation-byte run
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys("http://h/?%28(%A1=x")
    err = excinfo.value
    assert err.error == "invalid_utf8"
    assert (err.start, err.end) == (14, 17)  # the %A1 triplet alone


@pytest.mark.parametrize(
    "query,expected",
    [
        ("%", "invalid_percent_encoding"),
        ("%4", "invalid_percent_encoding"),
        ("%zz", "invalid_percent_encoding"),
        ("%4g", "invalid_percent_encoding"),
        ("a%", "invalid_percent_encoding"),
        ("a%2", "invalid_percent_encoding"),
        ("%C3", "invalid_utf8"),        # lead byte with no continuation
        ("%E2%82", "invalid_utf8"),     # truncated three-byte sequence
        ("%ff", "invalid_utf8"),        # never valid as a UTF-8 lead byte
    ],
)
def test_malformed_or_invalid_encoding(query, expected):
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys(f"http://h/?{query}=1")
    assert excinfo.value.error == expected


def test_invalid_percent_span_covers_the_short_unit():
    url = "http://h/?a%4=1"
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys(url)
    err = excinfo.value
    assert (err.start, err.end) == (11, 13)
    assert url[err.start:err.end] == "%4"

    url = "http://h/?%"
    with pytest.raises(QueryKeyParseError) as excinfo:
        audit_query_keys(url)
    err = excinfo.value
    assert (err.start, err.end) == (10, 11)
    assert url[err.start:err.end] == "%"


def test_invalid_encoding_in_a_value_is_not_an_error():
    # values take no part in decoding: malformed % and invalid UTF-8 pass
    report = audit_query_keys("http://h/?a=%zz&b=%c3%28&c=%ff&a=2")
    (group,) = report.collisions
    assert group.normalised == "a"


# ----------------------------------------------------------------- NFKC --

def test_nfkc_unifies_ascii_fullwidth_and_percent_encoded_keys():
    url = "https://e.test/p?name=a&%6eame=b&ｎame=c&name=d&safe=1#frag"
    report = audit_query_keys(url)
    (group,) = report.collisions
    assert group.normalised == "name"
    assert [m.index for m in group.members] == [1, 2, 3, 4]
    assert [m.raw_key for m in group.members] == [
        "name", "%6eame", "ｎame", "name",
    ]
    assert group.members[2].decoded_key == "ｎame"  # decode != normalise


def test_case_differences_are_not_collisions():
    report = audit_query_keys("http://h/?Name=1&name=2&NAME=3")
    assert report.collisions == []


def test_distinct_prefixes_do_not_collide():
    report = audit_query_keys("http://h/?name=1&names=2&username=3")
    assert report.collisions == []


def test_identical_raw_keys_collide():
    report = audit_query_keys("http://h/?id=1&id=2&id=3")
    (group,) = report.collisions
    assert len(group.members) == 3
    assert {m.raw_key for m in group.members} == {"id"}


def test_collision_groups_are_ordered_by_first_member_parameter():
    # groups z (params 1,3) before a (params 2,4)
    report = audit_query_keys("http://h/?z=1&a=2&z=3&a=4")
    assert [g.normalised for g in report.collisions] == ["z", "a"]


def test_members_are_in_parameter_order():
    report = audit_query_keys("http://h/?x=0&x=1&x=2")
    assert [m.index for m in report.collisions[0].members] == [1, 2, 3]


def test_report_dict_shape():
    url = "http://h/?b=1&b=2"
    body = audit_query_keys(url).to_dict(url)
    assert body == {
        "stored": False,
        "url": url,
        "url_length": 17,
        "parameter_count": 2,
        "collision_count": 1,
        "collisions": [
            {
                "normalised": "b",
                "count": 2,
                "members": [
                    {
                        "index": 1,
                        "raw_key": "b",
                        "decoded_key": "b",
                        "normalised_key": "b",
                        "start": 10,
                        "end": 11,
                        "position": {"line": 1, "column": 11},
                    },
                    {
                        "index": 2,
                        "raw_key": "b",
                        "decoded_key": "b",
                        "normalised_key": "b",
                        "start": 14,
                        "end": 15,
                        "position": {"line": 1, "column": 15},
                    },
                ],
            }
        ],
    }


# ------------------------------------------------------------------ API --

def test_endpoint_reports_collision_group(client):
    resp = client.post(
        "/api/v1/urls/query-keys",
        json={"url": "https://e.test/p?name=1&%6eame=2&ｎame=3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stored"] is False
    assert body["parameter_count"] == 3
    assert body["collision_count"] == 1
    group = body["collisions"][0]
    assert group["normalised"] == "name"
    assert [m["index"] for m in group["members"]] == [1, 2, 3]
    # positions are original-URL coordinates
    first = group["members"][0]
    assert body["url"][first["start"]:first["end"]] == "name"


def test_endpoint_clean_url(client):
    resp = client.post(
        "/api/v1/urls/query-keys",
        json={"url": "https://h.test/path?a=1&b=2"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["collision_count"] == 0
    assert body["collisions"] == []
    assert body["parameter_count"] == 2


def test_endpoint_no_query_component(client):
    resp = client.post(
        "/api/v1/urls/query-keys", json={"url": "https://h.test/path"}
    )
    assert resp.status_code == 200
    assert resp.json()["parameter_count"] == 0


def test_endpoint_422_on_malformed_percent_with_offsets(client):
    url = "http://h/?a%zz=1"
    resp = client.post("/api/v1/urls/query-keys", json={"url": url})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_percent_encoding"
    assert (detail["start"], detail["end"]) == (11, 14)
    assert url[detail["start"]:detail["end"]] == "%zz"
    assert detail["line"] == 1
    assert detail["column"] == 12
    assert detail["end_position"] == {"line": 1, "column": 15}
    # no partial report fields leak out
    assert set(detail) == {
        "error", "start", "end", "line", "column",
        "end_position", "message",
    }


def test_endpoint_422_on_invalid_utf8_with_offsets(client):
    resp = client.post(
        "/api/v1/urls/query-keys",
        json={"url": "http://h/?%C3%28=1"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_utf8"
    assert (detail["start"], detail["end"]) == (10, 16)


def test_endpoint_malformed_key_emits_no_partial_report(client):
    # the earlier keys would collide, but the later bad key suppresses
    # everything: the error response contains no collision report at all
    resp = client.post(
        "/api/v1/urls/query-keys",
        json={"url": "http://h/?a=1&a=2&%zz=3"},
    )
    assert resp.status_code == 422
    assert "collisions" not in resp.json()


def test_endpoint_bad_encoding_in_value_passes(client):
    resp = client.post(
        "/api/v1/urls/query-keys",
        json={"url": "http://h/?a=%zz&a=%c3%28"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["collision_count"] == 1
    assert body["collisions"][0]["normalised"] == "a"


def test_endpoint_requires_url(client):
    resp = client.post("/api/v1/urls/query-keys", json={})
    assert resp.status_code == 422


def test_endpoint_rejects_empty_url(client):
    resp = client.post("/api/v1/urls/query-keys", json={"url": ""})
    assert resp.status_code == 422


def test_endpoint_is_read_only(client):
    before = client.get("/api/v1/audits").json()["total"]
    client.post(
        "/api/v1/urls/query-keys",
        json={"url": "http://h/?a=1&a=2&a=3"},
    )
    after = client.get("/api/v1/audits").json()["total"]
    assert before == after


def test_endpoint_listed_in_root_catalog(client):
    endpoints = client.get("/").json()["endpoints"]
    assert "POST /api/v1/urls/query-keys" in endpoints
