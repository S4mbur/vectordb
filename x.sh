cat > create_oracle_table_growth_dashboard.py <<'PY'
import json
import urllib.request
import urllib.error

KIBANA_URL = "http://localhost:5601"
INDEX = "oracle-table-size-growth"

DASHBOARD_ID = "oracle-table-growth-dashboard"

def vega_bar_spec(title, metric_label, metric_script, suffix=""):
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": title,
        "width": 760,
        "height": 300,
        "padding": 8,
        "autosize": {"type": "fit", "contains": "padding"},
        "data": [
            {
                "name": "rows",
                "url": {
                    "%context%": True,
                    "%timefield%": "@timestamp",
                    "index": INDEX,
                    "body": {
                        "size": 0,
                        "runtime_mappings": {
                            "segment_key": {
                                "type": "keyword",
                                "script": """
String owner = '';
String obj = '';

if (doc.containsKey('owner.keyword') && doc['owner.keyword'].size() != 0) {
  owner = doc['owner.keyword'].value;
}

if (doc.containsKey('object_name.keyword') && doc['object_name.keyword'].size() != 0) {
  obj = doc['object_name.keyword'].value;
}

if (doc.containsKey('segment_full_name.keyword') && doc['segment_full_name.keyword'].size() != 0) {
  emit(doc['segment_full_name.keyword'].value);
} else if (owner.length() > 0 && obj.length() > 0) {
  emit(owner + '.' + obj);
}
"""
                            },
                            "metric_value": {
                                "type": "double",
                                "script": metric_script
                            }
                        },
                        "aggs": {
                            "top_segments": {
                                "terms": {
                                    "field": "segment_key",
                                    "size": 10,
                                    "order": {
                                        "metric": "desc"
                                    }
                                },
                                "aggs": {
                                    "metric": {
                                        "max": {
                                            "field": "metric_value"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "format": {
                    "property": "aggregations.top_segments.buckets"
                },
                "transform": [
                    {
                        "type": "formula",
                        "as": "metric_value",
                        "expr": "datum.metric.value"
                    },
                    {
                        "type": "filter",
                        "expr": "datum.metric_value != null && isValid(datum.metric_value)"
                    }
                ]
            }
        ],
        "scales": [
            {
                "name": "yscale",
                "type": "band",
                "domain": {
                    "data": "rows",
                    "field": "key"
                },
                "range": "height",
                "padding": 0.18
            },
            {
                "name": "xscale",
                "type": "linear",
                "domain": {
                    "data": "rows",
                    "field": "metric_value"
                },
                "range": "width",
                "nice": True,
                "zero": True
            }
        ],
        "axes": [
            {
                "orient": "bottom",
                "scale": "xscale",
                "title": metric_label,
                "grid": True
            },
            {
                "orient": "left",
                "scale": "yscale",
                "labelLimit": 320
            }
        ],
        "marks": [
            {
                "type": "rect",
                "from": {
                    "data": "rows"
                },
                "encode": {
                    "enter": {
                        "y": {
                            "scale": "yscale",
                            "field": "key"
                        },
                        "height": {
                            "scale": "yscale",
                            "band": 1
                        },
                        "x": {
                            "scale": "xscale",
                            "value": 0
                        },
                        "x2": {
                            "scale": "xscale",
                            "field": "metric_value"
                        },
                        "tooltip": {
                            "signal": "datum.key + ': ' + format(datum.metric_value, ',.2f') + '" + suffix + "'"
                        }
                    }
                }
            },
            {
                "type": "text",
                "from": {
                    "data": "rows"
                },
                "encode": {
                    "enter": {
                        "y": {
                            "scale": "yscale",
                            "field": "key",
                            "band": 0.5
                        },
                        "x": {
                            "scale": "xscale",
                            "field": "metric_value",
                            "offset": 6
                        },
                        "align": {
                            "value": "left"
                        },
                        "baseline": {
                            "value": "middle"
                        },
                        "text": {
                            "signal": "format(datum.metric_value, ',.2f') + '" + suffix + "'"
                        }
                    }
                }
            }
        ]
    }

size_script = """
if (doc.containsKey('current_size_mb') && doc['current_size_mb'].size() != 0) {
  emit(doc['current_size_mb'].value);
} else if (doc.containsKey('size_mb') && doc['size_mb'].size() != 0) {
  emit(doc['size_mb'].value);
}
"""

daily_script = """
if (doc.containsKey('pct_change_1d') && doc['pct_change_1d'].size() != 0) {
  emit(doc['pct_change_1d'].value);
}
"""

weekly_script = """
if (doc.containsKey('pct_change_7d') && doc['pct_change_7d'].size() != 0) {
  emit(doc['pct_change_7d'].value);
}
"""

def visualization_obj(obj_id, title, spec):
    vis_state = {
        "title": title,
        "type": "vega",
        "params": {
            "spec": json.dumps(spec, indent=2)
        },
        "aggs": []
    }

    return {
        "type": "visualization",
        "id": obj_id,
        "attributes": {
            "title": title,
            "description": "",
            "version": 1,
            "visState": json.dumps(vis_state),
            "uiStateJSON": "{}",
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "query": {
                        "query": "",
                        "language": "kuery"
                    },
                    "filter": []
                })
            }
        },
        "references": []
    }

viz_size_id = "oracle-table-size-top10-bar"
viz_daily_id = "oracle-table-daily-growth-top10-bar"
viz_weekly_id = "oracle-table-weekly-growth-top10-bar"

objects = [
    visualization_obj(
        viz_size_id,
        "Top 10 Table Size MB",
        vega_bar_spec(
            "Top 10 Table Size MB",
            "Size MB",
            size_script,
            " MB"
        )
    ),
    visualization_obj(
        viz_daily_id,
        "Top 10 Daily Table Growth %",
        vega_bar_spec(
            "Top 10 Daily Table Growth %",
            "Daily growth %",
            daily_script,
            "%"
        )
    ),
    visualization_obj(
        viz_weekly_id,
        "Top 10 Weekly Table Growth %",
        vega_bar_spec(
            "Top 10 Weekly Table Growth %",
            "Weekly growth %",
            weekly_script,
            "%"
        )
    )
]

panels = [
    {
        "version": "9.4.2",
        "type": "visualization",
        "gridData": {
            "x": 0,
            "y": 0,
            "w": 24,
            "h": 15,
            "i": "1"
        },
        "panelIndex": "1",
        "embeddableConfig": {},
        "panelRefName": "panel_1"
    },
    {
        "version": "9.4.2",
        "type": "visualization",
        "gridData": {
            "x": 24,
            "y": 0,
            "w": 24,
            "h": 15,
            "i": "2"
        },
        "panelIndex": "2",
        "embeddableConfig": {},
        "panelRefName": "panel_2"
    },
    {
        "version": "9.4.2",
        "type": "visualization",
        "gridData": {
            "x": 0,
            "y": 15,
            "w": 48,
            "h": 15,
            "i": "3"
        },
        "panelIndex": "3",
        "embeddableConfig": {},
        "panelRefName": "panel_3"
    }
]

objects.append({
    "type": "dashboard",
    "id": DASHBOARD_ID,
    "attributes": {
        "title": "Oracle Table Space Growth",
        "description": "Oracle table size and growth dashboard generated from Logstash index.",
        "panelsJSON": json.dumps(panels),
        "optionsJSON": json.dumps({
            "useMargins": True,
            "syncColors": False,
            "syncCursor": True,
            "syncTooltips": False
        }),
        "version": 1,
        "timeRestore": False,
        "kibanaSavedObjectMeta": {
            "searchSourceJSON": json.dumps({
                "query": {
                    "query": "",
                    "language": "kuery"
                },
                "filter": []
            })
        }
    },
    "references": [
        {
            "name": "panel_1",
            "type": "visualization",
            "id": viz_size_id
        },
        {
            "name": "panel_2",
            "type": "visualization",
            "id": viz_daily_id
        },
        {
            "name": "panel_3",
            "type": "visualization",
            "id": viz_weekly_id
        }
    ]
})

payload = json.dumps(objects).encode("utf-8")

req = urllib.request.Request(
    KIBANA_URL + "/api/saved_objects/_bulk_create?overwrite=true",
    data=payload,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
)

try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode("utf-8"))
        print()
        print("Dashboard hazır:")
        print(KIBANA_URL + "/app/dashboards#/view/" + DASHBOARD_ID)
except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code)
    print(e.read().decode("utf-8"))
    raise
PY

python3 create_oracle_table_growth_dashboard.py
