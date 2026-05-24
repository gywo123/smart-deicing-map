"""제설차 경로 지도와 재생형 네비게이션 시뮬레이션 유틸리티."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import folium


DEFAULT_MAPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "maps"))
DEFAULT_SPREAD_RATE_KG_M2 = 0.03


def bearing_label(from_coord: tuple[float, float], to_coord: tuple[float, float]) -> str:
    """두 좌표 사이 이동 방향을 8방위 한글 라벨로 반환한다."""
    dx = to_coord[0] - from_coord[0]
    dy = to_coord[1] - from_coord[1]
    angle = math.degrees(math.atan2(dx, dy)) % 360.0
    if angle < 22.5 or angle >= 337.5:
        return "북"
    if angle < 67.5:
        return "북동"
    if angle < 112.5:
        return "동"
    if angle < 157.5:
        return "남동"
    if angle < 202.5:
        return "남"
    if angle < 247.5:
        return "남서"
    if angle < 292.5:
        return "서"
    return "북서"


def _headline_from_results(results: dict[str, Any] | None) -> dict[str, str]:
    if not results:
        return {
            "risk": "위험구간 커버 개선",
            "cacl2": "염화칼슘 절감",
            "pop": "유동인구 보호 개선",
            "efficiency": "비용 효율 개선",
        }

    baseline = results.get("baseline", {})
    ai = results.get("ai", {})
    improvement = results.get("improvement", {})
    cacl2_reduction = improvement.get("cacl2_reduction_pct", 0.0)
    return {
        "risk": f"{baseline.get('risk_coverage', 0) * 100:.1f}% → {ai.get('risk_coverage', 0) * 100:.1f}%",
        "cacl2": f"{cacl2_reduction:.1f}% 절감",
        "pop": f"{baseline.get('pop_coverage', 0) * 100:.1f}% → {ai.get('pop_coverage', 0) * 100:.1f}%",
        "efficiency": f"{baseline.get('efficiency', 0):.2f} → {ai.get('efficiency', 0):.2f}",
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _estimate_deicing_kg(row: Any) -> float:
    """도로별 살포량 컬럼이 없을 때도 기존 파이프라인 기준으로 kg를 계산한다."""
    area = _safe_float(row.get("area", 0.0))
    explicit_kg = _safe_float(row.get("deicing_kg"), default=-1.0)
    if explicit_kg >= 0:
        return explicit_kg

    for column in ("spread_rate_g_m2", "recommended_spread_rate_g_m2"):
        rate_g_m2 = _safe_float(row.get(column), default=-1.0)
        if rate_g_m2 >= 0:
            return area * rate_g_m2 / 1000.0

    rate_kg_m2 = _safe_float(row.get("spread_rate"), default=DEFAULT_SPREAD_RATE_KG_M2)
    return area * rate_kg_m2


def create_navigation_simulation_map(
    roads_gdf,
    vehicle_route_roads,
    vehicle_route_coords,
    vehicle_meta,
    results: dict[str, Any] | None = None,
    maps_dir: str | None = None,
) -> None:
    """차량별 VRP 처리 순서를 재생할 수 있는 Folium 지도를 저장한다."""
    if maps_dir is None:
        maps_dir = DEFAULT_MAPS_DIR
    os.makedirs(maps_dir, exist_ok=True)

    center = [roads_gdf["cent_lat"].mean(), roads_gdf["cent_lon"].mean()]
    sim_map = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    bounds: list[tuple[float, float]] = []
    route_data: list[dict[str, Any]] = []

    for vehicle_idx, (route_roads, route_coords, meta) in enumerate(
        zip(vehicle_route_roads, vehicle_route_coords, vehicle_meta),
        start=1,
    ):
        if len(route_roads) == 0 or len(route_coords) == 0:
            continue

        color = meta["color"]
        movement_coords = meta.get("movement_coords") or route_coords
        movement_latlon = [(coord[1], coord[0]) for coord in movement_coords]
        if len(movement_latlon) >= 2:
            bounds.extend(movement_latlon)
            folium.PolyLine(
                movement_latlon,
                color=color,
                weight=2,
                dash_array="8",
                opacity=0.5,
                tooltip=f"{meta['name']} 도로망 이동 경로",
            ).add_to(sim_map)

        road_group = folium.FeatureGroup(name=f"{meta['name']} 처리 도로", show=True)
        steps = []
        cumul_m = 0.0
        route_rows = list(route_roads.iterrows())
        for order, (_, row) in enumerate(route_rows, start=1):
            coords = list(row.geometry.coords)
            road_latlon = [(coord[1], coord[0]) for coord in coords]
            bounds.extend(road_latlon)
            direction = "종료"
            if order < len(route_coords):
                direction = bearing_label(route_coords[order - 1], route_coords[order])

            deicing_kg = _estimate_deicing_kg(row)
            folium.PolyLine(
                locations=road_latlon,
                color=color,
                weight=4,
                opacity=0.78,
                tooltip=(
                    f"{meta['name']} #{order}<br>"
                    f"LINK: {row['LINK_ID']}<br>"
                    f"다음 방향: {direction}<br>"
                    f"위험도: {row['risk']:.3f}<br>"
                    f"살포량: {deicing_kg:.1f}kg"
                ),
            ).add_to(road_group)

            coord_idx = min(order - 1, len(route_coords) - 1)
            cumul_m += float(row["LENGTH"])
            steps.append(
                {
                    "order": order,
                    "lat": float(route_coords[coord_idx][1]),
                    "lon": float(route_coords[coord_idx][0]),
                    "link_id": str(row["LINK_ID"]),
                    "risk": round(float(row["risk"]), 3),
                    "length_m": round(float(row["LENGTH"]), 1),
                    "lanes": int(row["LANES"]),
                    "deicing_kg": round(deicing_kg, 1),
                    "direction": direction,
                    "cumul_km": round(cumul_m / 1000.0, 2),
                }
            )

        road_group.add_to(sim_map)
        route_data.append(
            {
                "id": vehicle_idx,
                "name": meta["name"],
                "color": color,
                "roads": int(meta["roads"]),
                "distance_km": float(meta["distance_km"]),
                "time_hours": float(meta.get("time_hours", 0)),
                "load_pct": round(float(meta.get("load_pct", 0)) * 100, 1),
                "movementPath": [
                    {"lat": float(coord[1]), "lon": float(coord[0])}
                    for coord in movement_coords
                ],
                "steps": steps,
            }
        )

    if bounds:
        sim_map.fit_bounds(bounds)

    panel_html = """
    <div id="snow-nav-sim-panel">
      <div class="sim-title">AI 제설 네비 시뮬레이션</div>
      <div class="sim-subtitle">차량별 VRP 순서를 재생하면서 확인</div>
      <div class="sim-kpi">
        <span>위험커버 <b id="sim-risk"></b></span>
        <span>CaCl2 <b id="sim-cacl2"></b></span>
        <span>유동인구 <b id="sim-pop"></b></span>
        <span>효율 <b id="sim-efficiency"></b></span>
      </div>
      <div class="sim-controls">
        <button id="nav-play">재생</button>
        <button id="nav-pause">일시정지</button>
        <button id="nav-reset">처음</button>
        <select id="nav-speed">
          <option value="900">느리게</option>
          <option value="550" selected>보통</option>
          <option value="250">빠르게</option>
        </select>
      </div>
      <div class="sim-progress-row">
        <span id="nav-step-label">0 / 0</span>
        <input id="nav-step-slider" type="range" min="0" max="1" value="0">
      </div>
      <div class="sim-progress-track"><div id="nav-progress-bar"></div></div>
      <div id="nav-current-steps"></div>
      <div class="sim-note">지도 선은 실제 도로 링크만 표시하고, 차량 아이콘은 처리 도로 중심점 기준 작업 순서를 재생합니다.</div>
    </div>
    <style>
      #snow-nav-sim-panel {
        position: fixed; top: 14px; right: 14px; width: 360px;
        max-height: calc(100vh - 36px); overflow-y: auto; z-index: 1000;
        background: rgba(255,255,255,0.96); border: 1px solid #d8dee9;
        border-radius: 12px; box-shadow: 0 12px 30px rgba(15,23,42,0.18);
        padding: 14px; font-family: "Noto Sans KR", "Malgun Gothic", sans-serif; color: #172033;
      }
      #snow-nav-sim-panel .sim-title { font-size: 17px; font-weight: 800; }
      #snow-nav-sim-panel .sim-subtitle { margin-top: 3px; font-size: 12px; color: #5f6b7a; }
      #snow-nav-sim-panel .sim-kpi { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 12px 0; }
      #snow-nav-sim-panel .sim-kpi span { background: #f4f7fb; border: 1px solid #e6ebf2; border-radius: 8px; padding: 7px 8px; font-size: 11px; }
      #snow-nav-sim-panel .sim-kpi b { display: block; margin-top: 2px; font-size: 12px; color: #0f172a; }
      #snow-nav-sim-panel .sim-controls { display: flex; gap: 6px; margin-bottom: 10px; }
      #snow-nav-sim-panel button, #snow-nav-sim-panel select { border: 0; border-radius: 8px; padding: 8px 10px; font-weight: 700; font-size: 12px; cursor: pointer; }
      #snow-nav-sim-panel button { background: #172033; color: #fff; }
      #snow-nav-sim-panel button:nth-child(2) { background: #64748b; }
      #snow-nav-sim-panel button:nth-child(3) { background: #e8eef6; color: #172033; }
      #snow-nav-sim-panel select { background: #f4f7fb; color: #172033; }
      #snow-nav-sim-panel .sim-progress-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #475569; }
      #nav-step-slider { flex: 1; }
      #snow-nav-sim-panel .sim-progress-track { height: 7px; background: #e6ebf2; border-radius: 999px; overflow: hidden; margin: 8px 0 12px; }
      #nav-progress-bar { height: 100%; width: 0%; background: linear-gradient(90deg, #1565c0, #e53935, #2e7d32); }
      #nav-current-steps { display: grid; gap: 8px; }
      #nav-current-steps .vehicle-card { border-radius: 8px; background: #fbfdff; border: 1px solid #e6ebf2; padding: 9px 10px; font-size: 12px; line-height: 1.45; }
      #snow-nav-sim-panel .sim-note { margin-top: 10px; padding-top: 9px; border-top: 1px solid #e6ebf2; font-size: 11px; color: #64748b; line-height: 1.45; }
      .snow-nav-vehicle-icon { background: transparent; border: 0; }
    </style>
    """
    sim_map.get_root().html.add_child(folium.Element(panel_html))

    js = """
    <script>
    window.addEventListener('load', function() {
      const map = __MAP_NAME__;
      const routeData = __ROUTE_DATA__;
      const headline = __HEADLINE_DATA__;
      const maxSteps = Math.max(...routeData.map(v => Math.max(v.movementPath?.length || 0, v.steps.length)), 0);
      let currentStep = 0;
      let timer = null;
      let speedMs = 550;
      const markers = {};

      function makeVehicleIcon(vehicle) {
        return L.divIcon({
          className: 'snow-nav-vehicle-icon',
          html: `<div style="width:34px;height:34px;border-radius:50%;background:${vehicle.color};color:white;border:3px solid white;box-shadow:0 5px 16px rgba(0,0,0,0.28);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px;">${vehicle.id}</div>`,
          iconSize: [34, 34],
          iconAnchor: [17, 17]
        });
      }

      routeData.forEach(vehicle => {
        if (!vehicle.steps.length && !(vehicle.movementPath || []).length) return;
        const first = (vehicle.movementPath || [])[0] || vehicle.steps[0];
        markers[vehicle.id] = L.marker([first.lat, first.lon], {
          icon: makeVehicleIcon(vehicle),
          zIndexOffset: 500
        }).addTo(map).bindPopup(`${vehicle.name} 출발`);
      });

      function setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
      }

      function render() {
        if (!maxSteps) return;
        const progressPct = maxSteps <= 1 ? 100 : (currentStep / (maxSteps - 1)) * 100;
        setText('nav-step-label', `${currentStep + 1} / ${maxSteps}`);
        document.getElementById('nav-progress-bar').style.width = `${progressPct}%`;
        document.getElementById('nav-step-slider').value = currentStep;

        document.getElementById('nav-current-steps').innerHTML = routeData.map(vehicle => {
          const path = vehicle.movementPath || [];
          const pathIdx = Math.min(currentStep, Math.max(path.length - 1, 0));
          const serviceRatio = maxSteps <= 1 ? 1 : currentStep / (maxSteps - 1);
          const idx = Math.min(Math.floor(serviceRatio * vehicle.steps.length), vehicle.steps.length - 1);
          const step = vehicle.steps[Math.max(idx, 0)];
          const markerPos = path[pathIdx] || step;
          const done = currentStep >= Math.max(path.length, vehicle.steps.length) - 1;
          if (markers[vehicle.id]) {
            markers[vehicle.id].setLatLng([markerPos.lat, markerPos.lon]);
            markers[vehicle.id].setPopupContent(`${vehicle.name} #${step.order}<br>LINK ${step.link_id}<br>위험도 ${step.risk}<br>${done ? '경로 완료' : '다음 방향 ' + step.direction}`);
          }
          return `<div class="vehicle-card" style="border-left:5px solid ${vehicle.color};"><strong style="color:${vehicle.color};">${vehicle.name}</strong> <span>${done ? '완료' : '#' + step.order}</span><br>LINK ${step.link_id} · 위험도 ${step.risk} · ${step.length_m}m<br>살포 ${step.deicing_kg}kg · 처리누적 ${step.cumul_km}km · 다음 ${step.direction}</div>`;
        }).join('');
      }

      function pause() {
        if (timer) clearInterval(timer);
        timer = null;
      }
      function play() {
        if (timer || !maxSteps) return;
        timer = setInterval(() => {
          if (currentStep >= maxSteps - 1) {
            pause();
            return;
          }
          currentStep += 1;
          render();
        }, speedMs);
      }

      document.getElementById('nav-play')?.addEventListener('click', play);
      document.getElementById('nav-pause')?.addEventListener('click', pause);
      document.getElementById('nav-reset')?.addEventListener('click', () => { pause(); currentStep = 0; render(); });
      document.getElementById('nav-speed')?.addEventListener('change', (event) => {
        speedMs = Number(event.target.value);
        if (timer) { pause(); play(); }
      });
      const slider = document.getElementById('nav-step-slider');
      if (slider) {
        slider.max = Math.max(maxSteps - 1, 0);
        slider.addEventListener('input', (event) => {
          pause();
          currentStep = Number(event.target.value);
          render();
        });
      }

      setText('sim-risk', headline.risk);
      setText('sim-cacl2', headline.cacl2);
      setText('sim-pop', headline.pop);
      setText('sim-efficiency', headline.efficiency);
      render();
    });
    </script>
    """
    js = (
        js.replace("__MAP_NAME__", sim_map.get_name())
        .replace("__ROUTE_DATA__", json.dumps(route_data, ensure_ascii=False))
        .replace("__HEADLINE_DATA__", json.dumps(_headline_from_results(results), ensure_ascii=False))
    )
    sim_map.get_root().html.add_child(folium.Element(js))
    folium.LayerControl(collapsed=True).add_to(sim_map)

    sim_map.save(os.path.join(maps_dir, "map_navigation_simulation.html"))
    print("  → map_navigation_simulation.html 저장 (재생형 네비 시뮬레이션)")
