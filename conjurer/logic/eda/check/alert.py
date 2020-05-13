from datetime import datetime

UB_MISSING_RATIO=0.5
UB_UNIQUE_COUNT=1000


def alert_columns(stat_df):
    alert_list = []
    for index, record in stat_df.iterrows():
        alert_list.extend(_check_missing_ratio(record))
        if record["adt"] == "datetime": alert_list.extend(_check_datetime_range(record))
        if record["adt"] == "category": alert_list.extend(_check_unique_count(record))
        if record["adt"] == "numeric": alert_list.extend(_check_outliers(record))
    for alert in alert_list:
        display(alert)


def _check_unique_count(record):
    alert_list = [ _generate_alert_message(
                    "WARN", record["column_name"], "too many unique values {0}".format(record["unique_count"]))] \
                            if record["unique_count"] > UB_UNIQUE_COUNT else \
                            [_generate_alert_message("WARN", record["column_name"], "only single unique value")] \
                                    if record["unique_count"] == 1 else []
    return alert_list


def _check_datetime_range(record):
    alert_list = []
    if record["min"] < datetime.datetime(1900, 1, 1):
        alert_list.append(_generate_alert_message(
                "ERROR", record["column_name"], "too old timestamp value {0}".format(record["min"])))
    if record["max"] > datetime.datetime(2999, 12, 31):
        alert_list.append(_generate_alert_message(
                "ERROR", record["column_name"], "too far future timestamp value {0}".format(record["min"])))
    return alert_list 


def _check_missing_ratio(record):
    alert_list = [_generate_alert_message(
            "WARN", record["column_name"], "too many missing values {0}".format(record["ratio_na"]))] \
                    if record["ratio_na"] > UB_MISSING_RATIO else []
    return alert_list


def _check_outliers(record):
    return []


def _generate_alert_message(error_type, column, message):
    return "[{0}] column {1} : {2}".format(error_type, column, message)
