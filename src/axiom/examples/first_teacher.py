"""Prepare an actual teacher request, then resume with an externally authored reply."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from axiom import Axiom
from axiom.benchmarks.run import write_json
from axiom.core.models import Demonstration, new_id
from axiom.core.recovery_models import RecoveryState
from axiom.core.recovery import validate_journal
from axiom.core.recovery_models import RecoveryConfig
from axiom.environments.classroom import ClassroomMenu
from axiom.environments.interaction import allowed_actions
from axiom.environments.menu import execute_method, semantic_guess
from axiom.learning.recovery import RecoveryLinear, recovery_features
from axiom.tools.teacher_bridge import TeacherReply, TeacherRequest, question_for


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path, data):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,allow_nan=False,indent=2)


def visible_state(env):
    return RecoveryState(base=env.observe(),spent=env.spent,opened=tuple(sorted(env.opened)),
        help_used=tuple(sorted(env.help_used)),success=env.success)


def freeze_tests(folder, namespace):
    """Before any teacher answer, seal 300 fixtures without displaying their contents."""
    rows = []
    for group in range(5):
        for regime in ("semantic","scrambled","legacy"):
            for i in range(20):
                seed = int(hashlib.sha256(f"{namespace}:{group}:{i}".encode()).hexdigest()[:16],16)
                env = ClassroomMenu(seed,regime=regime,allow_ask=False)
                rows.append(dict(group=group,regime=regime,index=i,case=env.export_private_case()))
    write_json(folder/"sealed-tests.json",rows)
    manifest = dict(sha256=sha(folder/"sealed-tests.json"),tasks=len(rows),created_utc=now(),namespace=namespace,
        scope="5 task-seed groups, one live teacher lesson; not five independent learner trainings")
    write_json(folder/"test-manifest.json",manifest)
    return manifest


def prepare_lesson(output, policy_state, *, seed=101, budget=4, allow_ask=True,
                   test_namespace="development", selector_provenance=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    policy = RecoveryLinear()
    policy.restore(policy_state)
    output.mkdir(parents=True,exist_ok=True)
    manifest = freeze_tests(output,test_namespace)
    env = ClassroomMenu(seed,split="lesson",budget=budget,allow_ask=allow_ask)
    chooser_log = []
    while not visible_state(env).done:
        state = visible_state(env)
        scores = policy.scores(recovery_features(state,{}))
        candidates = sorted([dict(method=a,score=scores[a]) for a in allowed_actions(state,{})],
            key=lambda c:(-c["score"],c["method"]))
        chosen = candidates[0]["method"]
        chooser_log.append(dict(state=state.model_dump(mode="json"),candidates=candidates,chosen=chosen))
        if chosen == "ask":
            request = TeacherRequest(request_id=new_id(),state=state,known_methods=("traverse",),
                question=question_for(state),candidates=tuple(candidates),chooser_digest=policy.digest,
                decision_reason="ask has the largest learned score among authorized, affordable candidates; no semantic skill exists yet")
            env.reserve_teacher()
            write_json(output/"request.json",request.model_dump(mode="json"))
            session = dict(seed=seed,budget=budget,allow_ask=allow_ask,request_id=request.request_id,
                request_sha256=sha(output/"request.json"),selector=policy.state(),
                opened=list(state.opened),trace=env.trace,spent=env.spent,created_utc=now(),
                test_sha256=manifest["sha256"],chooser_log=chooser_log,
                selector_provenance=selector_provenance or {"source":"host-supplied coefficients"})
            learner = Axiom(output/"learner.db")
            learner.observe(state.base)
            learner.save(output/"before-checkpoint.json")
            session.update(before_digest=learner.parameter_digest,
                before_checkpoint_sha256=sha(output/"before-checkpoint.json"))
            write_json(output/"session.json",session)
            learner.close()
            return dict(status="pending_teacher",request=request.model_dump(),spent=env.spent)
        if chosen != "traverse":
            raise AssertionError("no semantic skill or search channel should be available")
        env.open_menu(next(m.id for m in state.base.menus if m.id not in state.opened))
    result = dict(status="finished_without_teacher",success=env.success,spent=env.spent,
        trace=env.trace,chooser_log=chooser_log,teacher_calls=0)
    write_json(output/"no-request.json",result)
    return result


def resume_lesson(output, reply, *, teacher_source):
    output = Path(output)
    if (output/"lesson.json").exists():
        raise ValueError("this lesson has already been completed")
    if (output/"resume-started.json").exists():
        raise ValueError("partial session is locked; preserve its evidence and recover explicitly before retrying")
    session = read(output/"session.json")
    if sha(output/"request.json") != session["request_sha256"]:
        raise ValueError("pending request changed")
    if sha(output/"sealed-tests.json") != session["test_sha256"]:
        raise ValueError("frozen tests changed before teaching")
    if sha(output/"before-checkpoint.json") != session["before_checkpoint_sha256"]:
        raise ValueError("before checkpoint changed")
    request = TeacherRequest.model_validate(read(output/"request.json"))
    reply = TeacherReply.model_validate(reply)
    reply.validate_for(request)
    env = ClassroomMenu(session["seed"],split="lesson",budget=session["budget"],allow_ask=session["allow_ask"])
    # Reconstruct only already executed observations, preserving their actual order.
    for event in session["trace"]:
        if event.startswith("open:"):
            env.open_menu(event.split(":")[1])
    if visible_state(env) != request.state:
        raise ValueError("pending visible state does not match replay")
    env.reserve_teacher()
    write_new_json(output/"resume-started.json",dict(request_id=request.request_id,
        reply=reply.model_dump(mode="json"),teacher_source=teacher_source,started_utc=now(),
        note="An incomplete session remains locked; this marker never grants a retry or resets parameters."))
    learner = Axiom(output/"learner.db")
    try:
        if learner.store.events() != read(output/"before-checkpoint.json")["events"]:
            raise ValueError("learner differs from the bound before checkpoint")
        return _apply_reply(output,session,request,reply,teacher_source,env,learner)
    finally:
        learner.close()


def _apply_reply(output, session, request, reply, teacher_source, env, learner):
    before = learner.parameter_digest
    before_choice = learner.suggest()[0].method
    write_new_json(output/"teacher-reply.json",reply.model_dump(mode="json"))
    write_new_json(output/"teacher-provenance.json",dict(source=teacher_source,received_utc=now(),
        request_sha256=session["request_sha256"],reply_sha256=sha(output/"teacher-reply.json"),
        test_sha256=session["test_sha256"],note="teacher data are untrusted until simulator execution"))
    target = semantic_guess(env.observe(),reply.rule)
    if target is not None and target not in env.opened:
        env.open_menu(target)
    accepted = env.success
    recorded, recorded_choice = before, before_choice
    receipt = None
    if accepted:
        reward = 1 - .6*env.spent/env.observe().budget
        demo = Demonstration(observation=request.state.base,rule=reply.rule,verified=True,
            reward=reward,trace=tuple(env.trace),source=teacher_source,subject_id="external-teacher")
        learner.teach(demo)
        recorded = learner.parameter_digest
        recorded_choice = learner.suggest()[0].method
        learner.save(output/"recorded-checkpoint.json")
        receipt = learner.learn().model_dump()
        write_json(output/"verified-demonstration.json",demo.model_dump(mode="json"))
    else:
        # No skill/parameter promotion. Remaining old traversal can still run.
        for menu in env.observe().menus:
            if env.success or env.spent >= env.observe().budget:
                break
            if menu.id not in env.opened:
                env.open_menu(menu.id)
    after = learner.parameter_digest
    after_choice = learner.suggest()[0].method
    learner.save(output/"trained-checkpoint.json")
    learner.store.export_jsonl(output/"events.jsonl")
    if accepted:
        learner.rollback()
        rollback_ok = learner.parameter_digest == before
        learner.learn()
        if learner.parameter_digest != after:
            raise AssertionError("replaying the same verified lesson changed its fit")
    else:
        rollback_ok = None
    result = dict(accepted=accepted,success=env.success,operations=env.spent,trace=env.trace,
        teacher_calls=1,search_calls=0,before_digest=before,recorded_digest=recorded,
        trained_digest=after,before_choice=before_choice,recorded_choice=recorded_choice,
        after_choice=after_choice,learning=receipt,traversal_retained="traverse" in learner.skills,
        exact_rollback=rollback_ok,test_sha256=session["test_sha256"],completed_utc=now(),
        checkpoints={name:sha(output/name) for name in
            (["before-checkpoint.json","recorded-checkpoint.json","trained-checkpoint.json"] if accepted else
             ["before-checkpoint.json","trained-checkpoint.json"])},
        limitation="One live lesson; question language is templated, recovery chooser was pretrained on simulated experience.")
    write_json(output/"lesson.json",result)
    return result


def evaluate_transfer(output, *, tasks_per_cell=20):
    output = Path(output)
    if (output/"transfer-summary.json").exists():
        raise ValueError("evaluation already recorded")
    lesson = read(output/"lesson.json")
    if not lesson["accepted"]:
        raise ValueError("no validated lesson; retain failure log instead of fabricating transfer")
    if not 1 <= tasks_per_cell <= 20 or sha(output/"sealed-tests.json") != lesson["test_sha256"]:
        raise ValueError("invalid test count or changed frozen fixtures")
    if any(sha(output/name) != expected for name,expected in lesson["checkpoints"].items()):
        raise ValueError("lesson checkpoint changed before evaluation")
    trained = Axiom.load(output/"trained-checkpoint.json",output/"evaluation-learned.db")
    frozen = Axiom.load(output/"recorded-checkpoint.json",output/"evaluation-frozen.db")
    before = trained.parameter_digest
    frozen_before = frozen.parameter_digest
    if before != lesson["trained_digest"] or frozen_before != lesson["recorded_digest"]:
        trained.close()
        frozen.close()
        raise ValueError("checkpoint parameters differ from the observed lesson")
    rule = trained.rule
    records = []
    for item in read(output/"sealed-tests.json"):
        if item["index"] >= tasks_per_cell:
            continue
        for name in ("traversal","same_record_frozen","rule_memory","learned"):
            env = ClassroomMenu.from_private_case(item["case"])
            if name == "traversal":
                method = "traverse"
            elif name == "rule_memory":
                method = "semantic" if semantic_guess(env.observe(),rule) is not None else "traverse"
            else:
                agent = trained if name == "learned" else frozen
                agent.observe(env.observe())
                method = agent.suggest()[0].method
            outcome = execute_method(env,method,rule).model_dump(mode="json")
            records.append(dict(agent=name,group=item["group"],regime=item["regime"],
                task_id=env.observe().task_id,method=method,**outcome))
    def aggregate(rows):
        return dict(n=len(rows),**{k:float(np.mean([r[k] for r in rows]))
            for k in ("success","operations","reward","teacher_calls","search_calls","wrong_transfer","fallback_used")})
    lesson_id = read(output/"request.json")["state"]["base"]["task_id"]
    report = dict(classification="Single live-teacher pilot; frozen task variation, not replicated teacher training",
        metrics={name:aggregate([r for r in records if r["agent"]==name]) for name in {r["agent"] for r in records}},
        cells=[dict(agent=name,regime=regime,**aggregate([r for r in records if r["agent"]==name and r["regime"]==regime]))
            for name in ("traversal","same_record_frozen","rule_memory","learned") for regime in ("semantic","scrambled","legacy")],
        integrity=dict(evaluation_parameters_unchanged=trained.parameter_digest==before and frozen.parameter_digest==frozen_before,
            train_test_overlap=int(lesson_id in {r["task_id"] for r in records}),test_sha256=lesson["test_sha256"]),
        limits=["Developer and live teacher are the same assistant; this is not a double-blind independent study.",
            "Fixtures were sealed before the request, not inspected by the teacher, and not tuned after evaluation.",
            "Tests cover new layouts/function names within a known audio topic and label family.",
            "Same-record memory may match learned selection; parameter update alone does not prove unique benefit."])
    write_json(output/"transfer-summary.json",report)
    with (output/"transfer-episodes.jsonl").open("w",encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
    trained.close()
    frozen.close()
    return report


def load_selector_state(path):
    checkpoint = read(path)
    if checkpoint.get("schema_version") != "recovery-1":
        raise ValueError("requires a recovery checkpoint")
    config = RecoveryConfig.model_validate(checkpoint["config"])
    if config != RecoveryConfig():
        raise ValueError("classroom requires default feature, rule and inquiry configuration")
    events = checkpoint["events"]
    validate_journal(events,config.model_dump(mode="json"))
    policy, learned_ids = RecoveryLinear(), []
    for event in events:
        if not event["active"]:
            continue
        if event["kind"] in {"recovery_learn","recovery_rollback"}:
            policy.restore(event["payload"]["after"])
            learned_ids = event["payload"]["learned_ids"]
        elif event["kind"] == "recovery_reset":
            policy, learned_ids = RecoveryLinear(), []
    if not learned_ids:
        raise ValueError("checkpoint has no active learned recovery state")
    return policy.state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","resume","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--selector-checkpoint",type=Path)
    parser.add_argument("--reply",type=Path)
    parser.add_argument("--teacher-source",default="manual-external-teacher")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.selector_checkpoint is None:
            parser.error("prepare requires --selector-checkpoint")
        result = prepare_lesson(args.output,load_selector_state(args.selector_checkpoint),
            seed=0,test_namespace="first-live-lesson-v1",
            selector_provenance=dict(source="explicitly selected recovery checkpoint",
                checkpoint=str(args.selector_checkpoint),sha256=sha(args.selector_checkpoint)))
        if result["status"] == "pending_teacher":
            result["request"] = TeacherRequest.model_validate(result["request"]).model_dump(mode="json")
    elif args.command == "resume":
        if args.reply is None:
            parser.error("resume requires --reply")
        result = resume_lesson(args.output,read(args.reply),teacher_source=args.teacher_source)
    else:
        result = evaluate_transfer(args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
