"""Focused smoke / rule tests using isolated synthetic optimizer inputs."""
import unittest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app, optimize_lineup, OptimizeRequest, PlayerInput
from app.services.live_predictions import api_history_to_raw, explain_prediction, OMITTED_FEATURES
from data import SUM_COLS, to_player_gw
from features import build_dataset, feature_columns

class ApplicationTests(unittest.TestCase):
    def test_health_route(self):
        response=TestClient(app).get("/api/health")
        self.assertEqual(response.status_code,200)
        self.assertIn("historical evaluation",response.json()["prediction_scope"])
        self.assertTrue(response.json()["live_model_available"])

    def test_historical_player_data_and_metrics_exist(self):
        response=TestClient(app).get("/api/players?limit=3")
        self.assertEqual(response.status_code,200)
        body=response.json()
        self.assertEqual(body["scope"],"Historical test data (2025-26); not current projections")
        self.assertTrue(body["items"])
        report=TestClient(app).get("/api/model-performance")
        self.assertEqual(report.status_code,200)
        self.assertTrue(any(row["model"]=="XGBoost" for row in report.json()["items"]))
        compatibility=TestClient(app).get("/api/model-compatibility")
        self.assertEqual(compatibility.status_code,200)
        self.assertEqual(compatibility.json()["model_features"],73)

    def test_saved_model_loads_and_infers_one_compatible_row(self):
        import joblib
        from app.main import OUT
        saved=joblib.load(OUT/"xgb_model.joblib")
        row={feature:0 for feature in saved["features"]}
        response=TestClient(app).post("/api/predict/prepared",json=[row])
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(len(response.json()["predictions"]),1)
        incomplete=TestClient(app).post("/api/predict/prepared",json=[{saved["features"][0]:0}])
        self.assertEqual(incomplete.status_code,422)

    def test_historical_replay_matches_recorded_prediction(self):
        response=TestClient(app).get("/api/predictions/historical-inference?limit=2")
        self.assertEqual(response.status_code,200,response.text)
        rows=response.json()["items"]
        self.assertEqual(len(rows),2)
        self.assertTrue(all(row["existing_test_prediction"] is not None for row in rows))
        self.assertLess(max(abs(row["inference_minus_recorded"]) for row in rows),1e-5)

    def test_transfer_zero_extra_budget_and_captain_doubling(self):
        outgoing={"id":1,"name":"Outgoing","team":"Alpha","position":"MID","price":6,"projected_points":4}
        incoming={"id":2,"name":"Incoming","team":"Beta","position":"MID","price":6,"projected_points":5}
        transfer=TestClient(app).post("/api/transfer-analysis",json={"outgoing":outgoing,"candidates":[incoming],"budget":0})
        self.assertEqual(transfer.status_code,200,transfer.text)
        self.assertEqual(transfer.json()["options"][0]["projected_gain"],1)
        captain=TestClient(app).post("/api/captain-recommendation",json=[outgoing,incoming])
        self.assertEqual(captain.status_code,200,captain.text)
        self.assertEqual(captain.json()["captain"]["captaincy_total_if_estimate_realized"],10)

    def test_optimizer_obeys_squad_and_lineup_rules(self):
        spec=[("GK",2),("DEF",5),("MID",5),("FWD",3)]
        candidates=[]; player_id=1
        for position,count in spec:
            for j in range(count):
                candidates.append(PlayerInput(id=player_id,name=f"Test {player_id}",team=f"Club {(player_id-1)%5}",position=position,price=4.0,projected_points=float(2+player_id/10)))
                player_id+=1
        result=optimize_lineup(OptimizeRequest(players=candidates,budget=100))
        selected=result["selected"]
        self.assertEqual(len(selected),15)
        self.assertEqual(sum(p["starter"] for p in selected),11)
        self.assertEqual(sum(p["captain"] for p in selected),1)
        self.assertEqual(sum(p["vice_captain"] for p in selected),1)
        self.assertTrue(all(not p["vice_captain"] or p["starter"] for p in selected))
        self.assertEqual(sorted(p["bench_order"] for p in selected if not p["starter"] and p["position"]!="GK"),[1,2,3])
        self.assertEqual([p["bench_order"] for p in selected if not p["starter"] and p["position"]=="GK"],[0])
        self.assertLessEqual(result["cost"],100)
        defenders,midfielders,forwards=map(int,result["formation"].split("-"))
        self.assertEqual(defenders+midfielders+forwards,10)
        self.assertGreaterEqual(defenders,3); self.assertLessEqual(defenders,5)
        self.assertGreaterEqual(midfielders,2); self.assertLessEqual(midfielders,5)
        self.assertGreaterEqual(forwards,1); self.assertLessEqual(forwards,3)
        clubs={}
        for player in selected: clubs[player["team"]]=clubs.get(player["team"],0)+1
        self.assertLessEqual(max(clubs.values()),3)

    def test_optimizer_rejects_too_few_candidates(self):
        with self.assertRaises(HTTPException) as error:
            optimize_lineup(OptimizeRequest(players=[PlayerInput(id=1,name="Only",team="Club",position="GK",price=4,projected_points=1)]))
        self.assertEqual(error.exception.status_code,422)

    def test_original_research_optimizer_selects_legal_live_squad(self):
        import pandas as pd
        from optimize import pick_team
        players=[]
        spec=[("GK",0,2),("DEF",1,5),("MID",2,5),("FWD",3,3)]
        pid=1
        for position,code,count in spec:
            for i in range(count):
                players.append({"id":pid,"name":"Live "+str(pid),"team":"Club "+str((pid-1)%5),
                                "position":position,"pos_code":code,"price":4.0,
                                "predicted_points":float(pid),"projected_points":float(pid)})
                pid+=1
        selected=pick_team(pd.DataFrame(players),"predicted_points",budget=100)
        self.assertEqual(len(selected),15)
        self.assertEqual(int(selected.start.sum()),11)
        self.assertEqual(int(selected.captain.sum()),1)
        self.assertEqual(int(selected.vice_captain.sum()),1)
        self.assertTrue((selected.loc[selected.vice_captain==1,"start"]==1).all())
        self.assertLessEqual(selected.price.sum(),100)
        self.assertEqual(selected.groupby("team").size().max(),3)
        self.assertEqual(sorted(selected[selected.start==0].loc[selected.position!="GK","bench_order"].tolist()),[1,2,3])

    def test_api_event_mapping_preserves_reduced_training_features_and_order(self):
        import joblib
        import numpy as np
        from app.main import OUT
        original=joblib.load(OUT/"xgb_model.joblib")
        live=joblib.load(OUT/"xgb_live_model.joblib")
        expected=[c for c in original["features"] if c not in OMITTED_FEATURES]
        self.assertEqual(len(expected),70)
        self.assertEqual(live["features"],expected)
        teams={1:{"name":"Alpha"},2:{"name":"Beta"}}
        raw=[]; mapped=[]
        for i,(name,team_id,position) in enumerate([("One Player",1,"MID"),("Two Player",1,"DEF"),("Three Player",2,"MID"),("Four Player",2,"FWD")]):
            history=[]
            for gw in range(1,13):
                values={c:float((i+1)*(gw+1)%7) for c in SUM_COLS}
                values.update({"total_points":float((i+gw)%10),"minutes":90.0,"starts":1.0,
                               "expected_goals":.13*i+.01*gw,"expected_assists":.04*i+.02*gw,
                               "expected_goal_involvements":.1*gw,"expected_goals_conceded":.6+.1*i,
                               "value":50+i,"selected":1000,"xP":0.0})
                opponent=2 if team_id==1 else 1
                home=int((gw+i)%2==0)
                base={"name":name,"season":"2025-26","GW":gw,"team":teams[team_id]["name"],
                      "opp_name":teams[opponent]["name"],"position":position,"opponent_team":opponent,
                      "was_home":home,"opp_att_str":0.0,"opp_def_str":0.0,"opp_ovr_str":0.0,
                      "opp_strength_attack_home":1000.0,"opp_strength_attack_away":1000.0,
                      "opp_strength_defence_home":1000.0,"opp_strength_defence_away":1000.0,
                      "opp_strength_overall_home":1000.0,"opp_strength_overall_away":1000.0,**values}
                raw.append(base)
                event={"round":gw,"opponent_team":opponent,"was_home":bool(home),"value":50+i,"selected":1000,**values}
                history.append(event)
            player={"first_name":name.split()[0],"second_name":" ".join(name.split()[1:]),"team":team_id}
            converted=api_history_to_raw(history,player,teams)
            for row in converted:
                row.update({"season":"2025-26","position":position,"opp_att_str":0.0,"opp_def_str":0.0,"opp_ovr_str":0.0})
            mapped.extend(converted)
        offline=build_dataset(to_player_gw(__import__("pandas").DataFrame(raw)))
        from_api=build_dataset(to_player_gw(__import__("pandas").DataFrame(mapped)))
        key=["name","season","GW"]
        offline=offline.set_index(key).sort_index();from_api=from_api.set_index(key).sort_index()
        self.assertEqual([c for c in feature_columns(from_api.reset_index()) if c not in OMITTED_FEATURES],expected)
        np.testing.assert_allclose(offline[expected].to_numpy(),from_api[expected].to_numpy(),rtol=0,atol=1e-10,equal_nan=True)

    def test_explanation_reasons_always_include_measured_values_and_effects(self):
        import numpy as np
        import re
        import pandas as pd
        row=pd.Series({"minutes_last_5":88.0,"expected_assists_last_5":0.31})
        result=explain_prediction(row,np.array([0.44,-0.27]),["minutes_last_5","expected_assists_last_5"],{},"Example (ABC) vs Example FC (H), predicted 5.2 pts")
        self.assertEqual(len(result["reasons"]),2)
        self.assertTrue(all(re.search(r"\d",reason) for reason in result["reasons"]))
        self.assertTrue(all("model" in reason for reason in result["reasons"]))

    def test_fpl_client_serves_expired_cached_response_on_outage(self):
        import asyncio,time
        from datetime import datetime,timezone
        from unittest.mock import patch
        import httpx
        from app.services import fpl_client
        path="offline-cache-test/"
        stamp=datetime.now(timezone.utc)
        fpl_client._cache[path]=(time.monotonic()-1,{"items":[1,2]},stamp)
        class OfflineClient:
            async def get(self,url):
                raise httpx.ConnectError("offline")
        with patch.object(fpl_client,"_client",return_value=OfflineClient()):
            result=asyncio.run(fpl_client.get_json(path,ttl=1))
        self.assertEqual(result,{"items":[1,2]})
        self.assertTrue(fpl_client.cache_state(path)["stale"])
        self.assertEqual(fpl_client.cache_state(path)["last_success_at"],stamp.isoformat())

if __name__=="__main__": unittest.main()
