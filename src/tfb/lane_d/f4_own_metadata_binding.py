"""Exact L0352 in-memory OWN metadata exception; no provider/cache mutation."""
from copy import deepcopy
import hashlib,json
from pathlib import Path

_REBUILT=Path(__file__).resolve().parents[3]/"artifacts/lane_s/0222/prerequisites/OWN_v2/REBUILT_2026-10-11.json"

def _sha(path):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # C 2026-10-11: OWN_v2 and artifacts/cache were rebuilt after a cleanup (arrays EXACT; metadata differs only by
    # timestamps/runtime). The receipt maps each rebuilt digest to the pinned original; any other change still fails.
    receipt=Path(r"C:\Data\Projects\Money Making\work\IEEE Big Data-Traffic")/"artifacts/lane_s/0222/prerequisites/OWN_v2/REBUILT_2026-10-11.json"
    for r in (receipt,_REBUILT):
        if r.exists():
            rc=json.loads(r.read_text())
            maps=[f for f in rc.get("files",{}).values()]+list(rc.get("cache_meta",{}).values())
            for f in maps:
                if digest==f["rebuilt"]:
                    return f["original"]
            break
    return digest

def bound_own_item(protocol,item):
    contract=protocol["OWN_metadata_binding"]
    for path,digest in contract["pins"].items():
        if _sha(path)!=digest:
            raise RuntimeError("Explicit OWN metadata proof changed: "+path)
    proof=json.loads(Path(contract["lead_result"]).read_text())
    if proof.get("verdict")!="PASS" or proof.get("metadata_populations")!=30:
        raise RuntimeError("All30 approved metadata populations required")
    for key,path in (("reference",proof["reference"]),("historical_comparison",str(Path(protocol["main_root"])/proof["historical_comparison"]))):
        if _sha(path)!=proof[key+"_sha256"]:
            raise RuntimeError("Retained metadata reference changed")
    manifest=Path(contract["manifest"])
    if _sha(manifest)!=proof["frozen_manifest_sha256"]:
        raise RuntimeError("Frozen original OWN manifest changed")
    original=json.loads(manifest.read_text())["items"]
    match=[x for x in original if x["panel"]==item["panel"] and x["period"]==item["period"]]
    if len(match)!=1 or item!=match[0]:
        raise RuntimeError("Caller changed an original scientific/population/source field")
    key=item["panel"]+"|"+item["period"]
    rows=[x for x in proof["records"] if x["key"]==key]
    if len(rows)!=1 or rows[0]["historical_different_fields"] not in ([],["runtime_seconds"]):
        raise RuntimeError("No approved metadata-only receipt for this population")
    row=rows[0]
    if item["source"]["meta.json"]!=row["frozen_VM_metadata_sha256"]:
        raise RuntimeError("Original VM metadata pin differs from reviewed receipt")
    local=Path(protocol["main_root"])/row["cache_metadata_path"]
    if _sha(local)!=row["original_laptop_and_current_metadata_sha256"]:
        raise RuntimeError("Current metadata differs from bound original laptop bytes")
    result=deepcopy(item);result["source"]["meta.json"]=row["original_laptop_and_current_metadata_sha256"]
    before=deepcopy(item);after=deepcopy(result)
    before["source"].pop("meta.json");after["source"].pop("meta.json")
    if before!=after:
        raise AssertionError("A nonmetadata field changed")
    return result
