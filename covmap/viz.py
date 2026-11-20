"""Интерактивная HTML-карта (folium/Leaflet).

Слои: сетка по медиане RSRP, топ-3 худших, БС, кандидаты (отклонённые — с причиной),
выбранная пара с радиусом 1 км, покрытые красные квадраты, тепловая карта, день/ночь.
Панель «Что если» пересчитывает покрытие для любой пары площадок прямо в браузере.
"""

from __future__ import annotations

import html
import json

import folium
import pandas as pd
from folium.plugins import HeatMap

from .grid import RED

COLORS = {"green": "#1a9850", "yellow": "#f2c12e", "red": "#d73027"}
COLOR_NAMES = {"green": "хорошо", "yellow": "средне", "red": "слабо"}


def _cells_geojson(cells: pd.DataFrame) -> dict:
    features = []
    for r in cells.itertuples():
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [r.west, r.south], [r.east, r.south], [r.east, r.north], [r.west, r.north], [r.west, r.south],
                    ]],
                },
                "properties": {
                    "cell_id": r.cell_id,
                    "median": f"{r.median:.1f} дБм",
                    "count": int(r.count),
                    "color": r.color,
                    "label": COLOR_NAMES[r.color],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _cells_layer(cells: pd.DataFrame, name: str, show: bool = True, opacity: float = 0.55) -> folium.FeatureGroup:
    group = folium.FeatureGroup(name=name, show=show)
    if cells.empty:
        return group
    folium.GeoJson(
        _cells_geojson(cells),
        style_function=lambda f: {
            "fillColor": COLORS[f["properties"]["color"]],
            "color": COLORS[f["properties"]["color"]],
            "weight": 0.5,
            "fillOpacity": opacity,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#111"},
        tooltip=folium.GeoJsonTooltip(
            fields=["cell_id", "median", "count", "label"],
            aliases=["Квадрат", "Медиана RSRP", "Замеров", "Оценка"],
        ),
    ).add_to(group)
    return group


def _panel_html(details: dict) -> str:
    c = details["cleaning"]
    sel = details["selection"]
    bf, gr = sel["bruteforce"], sel["greedy"]
    colors = details["grid"]["colors"]
    dn = details["day_night"]
    worst_rows = "".join(
        f"<li><b>{html.escape(w['cell_id'])}</b> — {w['median_rsrp']:.1f} дБм ({w['count']} замеров)</li>"
        for w in details["worst_cells"]
    )
    rejected = details["candidates"]["rejected_near_tower"]
    rejected_rows = "".join(
        f"<li>{html.escape(r['id'])}: {r['distance_m']:.0f} м до {html.escape(str(r['nearest_tower']))}</li>"
        for r in rejected
    ) or "<li>нет</li>"
    greedy_note = (
        "совпадает с оптимумом"
        if sel["greedy_is_optimal"]
        else f"<b style='color:#d73027'>хуже на {bf['covered_weak_cells'] - gr['covered_weak_cells']}</b>"
    )
    return f"""
<div id="cm-panel">
  <div class="cm-head">
    <div class="cm-title">Карта покрытия · Бишкек</div>
    <button id="cm-toggle" title="Свернуть">–</button>
  </div>
  <div id="cm-body">
    <div class="cm-kpis">
      <div><span>{sel['weak_cells_before']}</span>слабых зон</div>
      <div><span>{sel['weak_cells_after']}</span>после 2 БС</div>
      <div><span>−{sel['weak_cells_reduction_pct']}%</span>слабых зон</div>
    </div>
    <div class="cm-legend">
      <i style="background:{COLORS['green']}"></i>≥ −90 ({colors['green']})
      <i style="background:{COLORS['yellow']}"></i>−105…−90 ({colors['yellow']})
      <i style="background:{COLORS['red']}"></i>&lt; −105 ({colors['red']})
    </div>

    <h4>Что если?</h4>
    <div class="cm-whatif">
      <select id="cm-a"></select><select id="cm-b"></select>
      <button id="cm-best">Оптимум</button>
    </div>
    <div id="cm-whatif-out"></div>

    <h4>Лучшая пара: {' + '.join(map(html.escape, bf['candidates'])) or '—'}</h4>
    <div class="cm-row">Полный перебор: {bf['covered_weak_cells']} квадратов, {bf['evaluated']} пар</div>
    <div class="cm-row">Жадный: {' + '.join(map(html.escape, gr['candidates'])) or '—'} → {gr['covered_weak_cells']}, {greedy_note}</div>

    <h4>Худшие квадраты</h4><ol>{worst_rows}</ol>

    <h4>Очистка: отброшено {c['dropped']} из {c['total_rows']}</h4>
    <div class="cm-row">нечисловые/пустые: {c['invalid_values']} · дубли: {c['duplicates']}</div>
    <div class="cm-row">RSRP вне [−140; −44]: {c['rsrp_out_of_range']} · вне района: {c['out_of_bounds']}</div>

    <h4>Отклонены (ближе 500 м к БС)</h4><ul>{rejected_rows}</ul>

    <h4>День ({dn['day_hours'][0]}–{dn['day_hours'][1]} ч) / ночь</h4>
    <div class="cm-row">красных: {dn['day']['red']} днём · {dn['night']['red']} ночью</div>
  </div>
</div>
<style>
#cm-panel {{position:absolute; z-index:1000; top:10px; left:56px; width:330px; max-height:calc(100vh - 20px);
  overflow:auto; background:rgba(255,255,255,.96); border-radius:10px; padding:12px 14px;
  font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:#1b1f24; box-shadow:0 4px 18px rgba(0,0,0,.18)}}
#cm-panel h4 {{margin:12px 0 4px; font-size:13px}}
#cm-panel ol, #cm-panel ul {{margin:2px 0; padding-left:18px}}
.cm-head {{display:flex; justify-content:space-between; align-items:center}}
.cm-title {{font-weight:700; font-size:15px}}
#cm-toggle {{border:0; background:#eef0f3; border-radius:6px; width:26px; height:24px; cursor:pointer}}
.cm-kpis {{display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin:10px 0}}
.cm-kpis div {{background:#f4f5f7; border-radius:8px; padding:6px; font-size:11px; color:#555; text-align:center}}
.cm-kpis span {{display:block; font-size:19px; font-weight:700; color:#1b1f24}}
.cm-legend {{font-size:12px}} .cm-legend i {{display:inline-block; width:11px; height:11px; border-radius:2px; margin:0 4px 0 8px; vertical-align:-1px}}
.cm-legend i:first-child {{margin-left:0}}
.cm-whatif {{display:flex; gap:6px}} .cm-whatif select {{flex:1; min-width:0}}
#cm-best {{border:0; background:#1a9850; color:#fff; border-radius:6px; padding:3px 8px; cursor:pointer}}
#cm-whatif-out {{margin-top:6px; padding:6px 8px; background:#f3effa; border-radius:6px}}
.cm-row {{color:#333}}
.cm-num {{background:#111; color:#fff; border-radius:50%; width:22px; height:22px; line-height:22px; text-align:center; font-weight:700; font:700 12px/22px system-ui}}
</style>
"""


def _whatif_script(map_name: str, a, details: dict) -> str:
    allowed = a.candidates[a.candidates["allowed"]]
    rejected = a.candidates[~a.candidates["allowed"]]
    data = {
        "radius": a.cfg.coverage_radius_m,
        "best": list(a.best.candidates),
        "optimum": a.best.covered_count,
        "candidates": [{"id": r.id, "lat": r.lat, "lon": r.lon} for r in allowed.itertuples()],
        "rejected": [{"id": r.id, "dist": round(float(r.nearest_tower_m))} for r in rejected.itertuples()],
        "red": [
            {"id": r.cell_id, "lat": r.center_lat, "lon": r.center_lon, "b": [[r.south, r.west], [r.north, r.east]]}
            for r in a.red_cells.itertuples()
        ],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""
window.addEventListener("load", function () {{
  var map = window["{map_name}"];
  var D = {payload};
  var layer = L.layerGroup().addTo(map);
  var selA = document.getElementById("cm-a"), selB = document.getElementById("cm-b");
  var out = document.getElementById("cm-whatif-out");

  function hav(lat1, lon1, lat2, lon2) {{
    var R = 6371008.8, r = Math.PI / 180;
    var dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
    var x = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(Math.min(1, x)));
  }}
  var byId = {{}};
  D.candidates.forEach(function (c) {{
    byId[c.id] = c;
    [selA, selB].forEach(function (s) {{ s.add(new Option(c.id, c.id)); }});
  }});
  D.rejected.forEach(function (c) {{
    [selA, selB].forEach(function (s) {{
      var o = new Option(c.id + " ✕ " + c.dist + " м до БС", c.id); o.disabled = true; s.add(o);
    }});
  }});

  function update() {{
    layer.clearLayers();
    var ids = [selA.value, selB.value].filter(function (v, i, arr) {{ return v && arr.indexOf(v) === i; }});
    var covered = {{}};
    ids.forEach(function (id) {{
      var c = byId[id];
      L.circle([c.lat, c.lon], {{radius: D.radius, color: "#6f42c1", weight: 2, dashArray: "6 4", fillOpacity: 0.04}})
        .bindTooltip("Что если: " + id).addTo(layer);
      D.red.forEach(function (cell) {{
        if (hav(c.lat, c.lon, cell.lat, cell.lon) <= D.radius) covered[cell.id] = cell;
      }});
    }});
    Object.keys(covered).forEach(function (k) {{
      L.rectangle(covered[k].b, {{color: "#6f42c1", weight: 2, fill: false}}).addTo(layer);
    }});
    var n = Object.keys(covered).length;
    var diff = D.optimum - n;
    out.innerHTML = "<b>" + ids.join(" + ") + "</b>: покрыто <b>" + n + "</b> из " + D.red.length +
      " красных квадратов" + (diff > 0 ? " · до оптимума не хватает " + diff : " · <b>это оптимум</b>");
  }}
  function setBest() {{
    if (D.best[0]) selA.value = D.best[0];
    if (D.best[1]) selB.value = D.best[1];
    update();
  }}
  selA.onchange = selB.onchange = update;
  document.getElementById("cm-best").onclick = setBest;
  document.getElementById("cm-toggle").onclick = function () {{
    var body = document.getElementById("cm-body");
    var hidden = body.style.display === "none";
    body.style.display = hidden ? "" : "none";
    this.textContent = hidden ? "–" : "+";
  }};
  setBest();
}});
"""


def build_map(a, details: dict) -> folium.Map:
    lat_c, lon_c = a.cfg.bounds.center
    if not a.measurements.empty:
        lat_c, lon_c = float(a.measurements["lat"].median()), float(a.measurements["lon"].median())
    m = folium.Map(location=[lat_c, lon_c], zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    _cells_layer(a.cells, "Покрытие: медиана RSRP").add_to(m)

    # Красные квадраты, которые закроют новые БС.
    covered_layer = folium.FeatureGroup(name="Закрываются новыми БС")
    covered_cells = a.red_cells[a.red_cells["cell_id"].isin(a.best.covered)]
    for r in covered_cells.itertuples():
        folium.Rectangle([[r.south, r.west], [r.north, r.east]], color="#0b6e2e", weight=2.5, fill=False,
                         tooltip=f"{r.cell_id}: закроется новой БС").add_to(covered_layer)
    covered_layer.add_to(m)

    worst_layer = folium.FeatureGroup(name="Топ-3 худших квадрата")
    for rank, r in enumerate(a.worst.itertuples(), start=1):
        folium.Rectangle([[r.south, r.west], [r.north, r.east]], color="#000", weight=3, fill=False).add_to(worst_layer)
        folium.Marker(
            [r.center_lat, r.center_lon],
            icon=folium.DivIcon(html=f'<div class="cm-num">{rank}</div>', icon_size=(22, 22), icon_anchor=(11, 11)),
            tooltip=f"#{rank} {r.cell_id}: {r.median:.1f} дБм, {r.count} замеров",
        ).add_to(worst_layer)
    worst_layer.add_to(m)

    towers_layer = folium.FeatureGroup(name="Существующие БС")
    for r in a.towers.itertuples():
        folium.Circle([r.lat, r.lon], radius=a.cfg.min_tower_distance_m, color="#334", weight=1, dash_array="3 4",
                      fill=False).add_to(towers_layer)
        folium.CircleMarker([r.lat, r.lon], radius=7, color="#fff", weight=2, fill=True, fill_color="#1f2a44",
                            fill_opacity=1, tooltip=f"БС {r.id}").add_to(towers_layer)
    towers_layer.add_to(m)

    cand_layer = folium.FeatureGroup(name="Площадки-кандидаты")
    best_ids = set(a.best.candidates)
    for r in a.candidates.itertuples():
        if r.id in best_ids:
            continue
        if r.allowed:
            n = len(a.coverage.get(r.id, ()))
            folium.CircleMarker([r.lat, r.lon], radius=6, color="#1565c0", fill=True, fill_opacity=0.9,
                                tooltip=f"{r.id}: покрывает {n} красных").add_to(cand_layer)
        else:
            folium.CircleMarker([r.lat, r.lon], radius=6, color="#888", fill=True, fill_color="#ccc", fill_opacity=0.9,
                                tooltip=f"{r.id}: отклонена — {r.nearest_tower_m:.0f} м до БС {r.nearest_tower}"
                                ).add_to(cand_layer)
    cand_layer.add_to(m)

    best_layer = folium.FeatureGroup(name="Новые БС (лучшая пара)")
    for r in a.candidates[a.candidates["id"].isin(best_ids)].itertuples():
        folium.Circle([r.lat, r.lon], radius=a.cfg.coverage_radius_m, color="#0b6e2e", weight=2,
                      fill=True, fill_opacity=0.06).add_to(best_layer)
        folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="star"),
                      tooltip=f"Новая БС {r.id}: покрывает {len(a.coverage.get(r.id, ()))} красных").add_to(best_layer)
    best_layer.add_to(m)

    if not a.measurements.empty:
        weakness = ((-a.measurements["rsrp_dbm"] - 80) / 40).clip(0.05, 1.0)
        heat = HeatMap(
            list(zip(a.measurements["lat"], a.measurements["lon"], weakness)),
            name="Тепловая карта слабого сигнала", radius=14, blur=18, min_opacity=0.25, show=False,
        )
        heat.add_to(m)

    _cells_layer(a.day_cells, f"День ({a.cfg.day_hours[0]}–{a.cfg.day_hours[1]} ч)", show=False).add_to(m)
    _cells_layer(a.night_cells, "Ночь", show=False).add_to(m)

    if not a.cells.empty:
        m.fit_bounds([[a.cells["south"].min(), a.cells["west"].min()], [a.cells["north"].max(), a.cells["east"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_panel_html(details)))
    m.get_root().script.add_child(folium.Element(_whatif_script(m.get_name(), a, details)))
    return m
