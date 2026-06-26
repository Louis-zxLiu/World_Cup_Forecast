from pathlib import Path

from worldcup_forecast.odds import FiveHundredLotteryProvider, implied_probabilities


def test_parse_sample_china_lottery_snapshot():
    html = Path("data/samples/china_lottery_sample.html").read_text(encoding="utf-8")
    records = FiveHundredLotteryProvider().parse(html)

    assert len(records) == 3
    assert records[0].match_id == "周五001"
    assert records[0].home_team == "Brazil"
    assert records[0].away_team == "Germany"
    assert records[0].play_type == "胜平负"
    assert records[0].win_odds == 2.25
    assert records[1].play_type == "让球胜平负"
    assert records[1].handicap == "-1"


def test_parse_500_jczq_rows_uses_data_attributes_not_page_buttons():
    html = Path("data/samples/500_jczq_sample.html").read_text(encoding="utf-8")
    records = FiveHundredLotteryProvider().parse(html)

    assert len(records) == 2
    assert records[0].match_id == "周五061"
    assert records[0].home_team == "Norway"
    assert records[0].away_team == "France"
    assert records[0].play_type == "胜平负"
    assert records[0].handicap is None
    assert records[0].win_odds == 4.50
    assert records[0].draw_odds == 4.15
    assert records[0].lose_odds == 1.50
    assert records[1].play_type == "让球胜平负"
    assert records[1].handicap == "1"
    assert all("亚" not in record.home_team for record in records)
    assert all("荐" not in record.away_team for record in records)


def test_implied_probabilities_remove_overround():
    html = Path("data/samples/china_lottery_sample.html").read_text(encoding="utf-8")
    record = FiveHundredLotteryProvider().parse(html)[0]
    probs = implied_probabilities(record)

    assert round(sum(value for value in probs.values() if value is not None), 6) == 1
    assert probs["home_win"] is not None
    assert probs["draw"] is not None
    assert probs["away_win"] is not None
