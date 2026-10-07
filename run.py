#!/usr/bin/env python3
"""Unified launcher for Omni Autonomy Next.

Interactive use:
    python3 run.py

Direct use:
    python3 run.py demo
    python3 run.py real
    python3 run.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")
WORKSPACE = ROOT / "ros2_ws"
WORKSPACE_SETUP = WORKSPACE / "install" / "setup.bash"
PACKAGE_SHARE = (
    WORKSPACE / "install" / "omni_autonomy_next" / "share" / "omni_autonomy_next"
)
GATEWAY = ROOT / "bacon_gateway"
GATEWAY_BUILD = GATEWAY / "build"
GATEWAY_BINARY = GATEWAY_BUILD / "omni_gateway_next"

ROS_SYSTEM_ACTIONS = {"demo", "real", "full"}
ROS_TOOL_ACTIONS = {
    "gui", "rviz", "goal", "remember", "remembered-goal", "cad-import",
    "check-odometry", "record-run",
    "diagnose-chain",
    "check-drive-directions",
}
GATEWAY_ACTIONS = {"gateway", "gateway-monitor", "gateway-build", "gateway-test"}
ACTIONS = (
    "menu",
    "demo",
    "real",
    "full",
    "gui",
    "rviz",
    "goal",
    "remember",
    "remembered-goal",
    "verify",
    "campaign",
    "autotune",
    "rl-train",
    "rl-evaluate",
    "ros-build",
    "cad-import",
    "check-odometry",
    "record-run",
    "diagnose-chain",
    "check-drive-directions",
    "gateway-build",
    "gateway-test",
    "gateway-monitor",
    "gateway",
)

MENU_ITEMS = (
    ("demo", "安全デモ（実機なし・モーターOFF）"),
    ("real", "センサー確認のみ（実機は動きません・スマホ受信OFF）"),
    ("full", "実機自動制御：スマホ指示・Jetson目的地（モーターON）"),
    ("gui", "速度・Arm・E-stop 操作GUIのみ"),
    ("rviz", "RVizのみ"),
    ("goal", "起動中システムへ設定済み目標IDを送信"),
    ("remember", "ロボットの現在地を名前付きで保存"),
    ("remembered-goal", "保存した名前の地点へ移動"),
    ("verify", "全ソフトウェア検証"),
    ("campaign", "モンテカルロ・キャンペーン"),
    ("autotune", "走行パラメーター自動調整"),
    ("rl-train", "安全制約付き強化学習（シミュレーション）"),
    ("rl-evaluate", "学習済み方策の未学習シード評価"),
    ("ros-build", "ROS 2 ワークスペースのビルド"),
    ("cad-import", "フィールドCADからYAMLを生成"),
    ("check-odometry", "計測輪オドメトリの符号・スケール確認（手押し・モーターOFF）"),
    ("record-run", "走行中の指令と応答を記録して蛇行の発生箇所を切り分け"),
    ("diagnose-chain", "走行中の指令チェーン全段の遅れ・振動・経路追従を計測"),
    ("check-drive-directions", "実機で +x/+y/+yaw の物理方向を確定（ACCEPTANCE step 3・モーターON）"),
    ("gateway-build", "bacon6ゲートウェイのビルド"),
    ("gateway-test", "bacon6ゲートウェイのテスト"),
    ("gateway-monitor", "ゲートウェイUDP安全監視（ハードウェアなし）"),
    ("gateway", "bacon6実機ゲートウェイ（安全確認必須）"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omni Autonomy Next の全機能を起動する統合ランチャーです。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=ACTIONS,
        help="起動する機能。省略時は対話メニューを表示します",
    )
    # Keep the first launcher's syntax working.
    parser.add_argument(
        "--mode",
        choices=("demo", "real"),
        help="互換オプション。通常は action に demo または real を指定してください",
    )
    parser.add_argument(
        "--motors",
        action="store_true",
        help="real と併用して full（実機モーターON）にします",
    )
    parser.add_argument(
        "--accept-motor-risk",
        action="store_true",
        help="車輪浮上試験完了後に full/gateway と併用する安全確認フラグです",
    )
    parser.add_argument("--no-gui", action="store_true", help="システム起動時にGUIを省略します")
    parser.add_argument("--no-rviz", action="store_true", help="システム起動時にRVizを省略します")
    # 既定は100Hz軌道追従器（2026-08-07 16:59 の構成に合わせて復帰）。
    # MPPI に戻すのは比較測定のとき。判断材料は system.launch.py の
    # tracker 引数のコメントにある。
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="100Hz軌道追従器の代わりに20HzのMPPIで走らせます（比較測定用）",
    )
    parser.add_argument("--build", action="store_true", help="ROS起動前に再ビルドします")
    parser.add_argument('--motion-mode', choices=['simultaneous', 'staged_heading'],
                        default='simultaneous', help='旋回を走行と同時に行うか、広い場所で分離するか')
    parser.add_argument(
        "--seconds", type=float, default=45.0,
        help="diagnose-chain の記録時間[秒]")
    parser.add_argument(
        "--navigation-delay",
        type=float,
        # 通常は局在器の準備完了を検出してこの値より早く起動する。これは
        # 準備完了ログを取得できない場合の安全側フォールバック期限。
        default=26.0,
        metavar="SECONDS",
        help="局在器の準備完了を検出できない場合のNav2起動待ち秒数",
    )
    parser.add_argument("--episodes", type=int, default=1200, help="campaign の総試行数")
    parser.add_argument(
        "--random-episodes", type=int, default=120, help="campaign のランダム試行数"
    )
    parser.add_argument("--seed", type=int, default=20260801, help="campaign の乱数シード")
    parser.add_argument(
        "--goal-id",
        type=str,
        default="",
        help="設定済み目標地点ID（0〜7）。demo/real/fullでは起動後に自動実行します",
    )
    parser.add_argument(
        "--pose-name",
        type=str,
        default="",
        help="remember / remembered-goal で保存または指定する地点名",
    )
    parser.add_argument(
        "--initial-pose-id",
        type=str,
        default="1",
        choices=tuple(str(value) for value in range(8)),
        help=(
            "起動時の自己位置ID。既定値1は添付された試合画像の "
            "x=-1.80, y=4.75, yaw=-90degです"
        ),
    )
    parser.add_argument(
        "--training-episodes", type=int, default=700, help="rl-train の学習試行数"
    )
    parser.add_argument(
        "--training-random-episodes",
        type=int,
        default=60,
        help="rl-train のランダム経路試行数",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=98, help="rl-train 後の評価試行数"
    )
    parser.add_argument(
        "--eval-random-episodes",
        type=int,
        default=0,
        help="rl-train 後のランダム評価試行数（既定は設定済み目標のみ）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "simulation" / "results" / "rl_policy.json",
        help="強化学習モデルの保存・読込先",
    )
    parser.add_argument("--input-stl", type=Path, help="cad-import の入力STL")
    parser.add_argument("--output-yaml", type=Path, help="cad-import の出力YAML")
    parser.add_argument("--slice-z-mm", type=float, default=130.0, help="CAD切断面の高さ")
    parser.add_argument("--simplify-mm", type=float, default=0.05, help="CAD線分の簡略化誤差")
    parser.add_argument(
        "--layout-config", type=Path, help="cad-import のレイアウト補正YAML"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="gateway の状態表示を有効にします"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="処理を実行せず、実行内容だけ表示します"
    )
    args = parser.parse_args()

    if args.navigation_delay < 0:
        parser.error("--navigation-delay は 0 以上を指定してください")
    if args.episodes <= 0:
        parser.error("--episodes は 1 以上を指定してください")
    if args.random_episodes < 0 or args.random_episodes > args.episodes:
        parser.error("--random-episodes は 0 以上かつ --episodes 以下にしてください")
    if args.training_episodes <= 0:
        parser.error("--training-episodes は 1 以上を指定してください")
    if not 0 <= args.training_random_episodes <= args.training_episodes:
        parser.error(
            "--training-random-episodes は 0 以上かつ --training-episodes 以下にしてください"
        )
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes は 1 以上を指定してください")
    if not 0 <= args.eval_random_episodes <= args.eval_episodes:
        parser.error(
            "--eval-random-episodes は 0 以上かつ --eval-episodes 以下にしてください"
        )
    if args.action and args.mode:
        parser.error("action と --mode は同時に指定できません")
    if args.motors and args.action not in (None, "real", "full"):
        parser.error("--motors は real または --mode real と一緒に指定してください")
    if args.goal_id and args.goal_id not in {str(value) for value in range(8)}:
        parser.error("--goal-id は設定済みの 0〜7 を指定してください")
    if args.action == "goal" and not args.goal_id:
        parser.error("goal には --goal-id を指定してください")
    if args.action in {"remember", "remembered-goal"} and not args.pose_name.strip():
        parser.error(f"{args.action} には --pose-name を指定してください")
    return args


def shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def choose_from_menu() -> tuple[str | None, bool]:
    print("\nOmni Autonomy Next - 機能を選択してください\n")
    print("スマホの地点指示・Jetsonから実機を動かす場合は 3 を選択してください。")
    print("1 は模擬走行、2 はセンサー確認、4 は操作画面のみです。\n")
    for index, (_, label) in enumerate(MENU_ITEMS, start=1):
        print(f"  {index:2}. {label}")
    print("   0. 終了")

    while True:
        try:
            answer = input("\n番号 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nキャンセルしました。")
            return None, False
        if answer == "0":
            return None, False
        if answer.isdigit() and 1 <= int(answer) <= len(MENU_ITEMS):
            action = MENU_ITEMS[int(answer) - 1][0]
            break
        print("一覧にある番号を入力してください。")

    accepted = False
    if action in {"full", "gateway"}:
        print("\n警告: この機能は実機モーターを動かす可能性があります。")
        print("docs/ACCEPTANCE.md の車輪浮上試験が完了している場合だけ続行してください。")
        try:
            accepted = input("続行するには ACCEPT と入力 > ").strip() == "ACCEPT"
        except (EOFError, KeyboardInterrupt):
            accepted = False
        if not accepted:
            print("安全のためキャンセルしました。")
            return None, False
    return action, accepted


def resolve_action(args: argparse.Namespace) -> str | None:
    action = args.action
    if action == "menu" or (action is None and args.mode is None and sys.stdin.isatty()):
        action, accepted = choose_from_menu()
        args.accept_motor_risk = args.accept_motor_risk or accepted
    elif action is None:
        # Preserve non-interactive compatibility with the original safe launcher.
        action = args.mode or "demo"

    if args.motors:
        if action != "real":
            raise RuntimeError("--motors は real と一緒に指定してください")
        action = "full"

    # check-drive-directions is the ACCEPTANCE step 3 tool itself, so it is a
    # motor action and carries the same interlock.
    motor_actions = {"full", "gateway", "check-drive-directions"}
    if action in motor_actions and not args.accept_motor_risk:
        raise RuntimeError(
            f"{action} は実機を動かす可能性があります。docs/ACCEPTANCE.md の試験後、"
            "--accept-motor-risk を付けて再実行してください。"
        )
    if args.accept_motor_risk and action not in motor_actions:
        raise RuntimeError(
            "--accept-motor-risk は full / gateway / check-drive-directions 専用です")
    return action


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"必要なコマンドが見つかりません: {name}")


def ensure_ros() -> None:
    if not ROS_SETUP.is_file():
        raise RuntimeError(
            f"ROS 2 Jazzy が見つかりません: {ROS_SETUP}\n"
            "ROS 2 Jazzy をインストールしてから再実行してください。"
        )
    if not WORKSPACE.is_dir():
        raise RuntimeError(f"ROS 2 ワークスペースが見つかりません: {WORKSPACE}")


def workspace_is_built() -> bool:
    plugin = WORKSPACE / "install" / "omni_route_bt" / "lib" / "libomni_remove_passed_bucket_goals_bt_node.so"
    return (WORKSPACE_SETUP.is_file() and PACKAGE_SHARE.is_dir() and plugin.is_file()
            and build_matches_sources(WORKSPACE / 'src', WORKSPACE / 'install'))


def source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    ignored = {'build', 'install', 'log', '__pycache__', '.pytest_cache', '.git'}

    def visit(directory: Path, ancestors: set[Path]) -> None:
        resolved = directory.resolve()
        if resolved in ancestors:
            raise RuntimeError(f'ソースのディレクトリリンクが循環しています: {directory}')
        for path in sorted(directory.iterdir()):
            if path.name in ignored:
                continue
            relative = path.relative_to(source)
            if path.is_symlink():
                digest.update(str(relative).encode('utf-8') + b'\0link\0')
                digest.update(os.readlink(path).encode('utf-8') + b'\0')
            if path.is_dir():
                visit(path, ancestors | {resolved})
            elif path.is_file():
                digest.update(str(relative).encode('utf-8') + b'\0')
                digest.update(hashlib.sha256(path.read_bytes()).digest())

    visit(source, set())
    return digest.hexdigest()


def build_matches_sources(source: Path, output: Path) -> bool:
    try:
        return (output / '.source.sha256').read_text().strip() == source_digest(source)
    except OSError:
        return False


def remember_built_sources(source: Path, output: Path, expected: str) -> None:
    # Never certify a build if an editor changed its inputs while it ran.
    if source_digest(source) != expected:
        raise RuntimeError('ビルド中にソースが変更されました。再ビルドしてください。')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / '.source.sha256.tmp'
    temporary.write_text(expected + '\n')
    temporary.replace(output / '.source.sha256')


def build_workspace(dry_run: bool = False) -> None:
    ensure_ros()
    require_program("colcon")
    parts = ["colcon", "build", "--symlink-install", "--packages-up-to", "omni_autonomy_next"]
    if dry_run:
        print(f"build: source {ROS_SETUP} && {shell_command(parts)}")
        return
    print("[setup] ROS 2 ワークスペースをビルドします...", flush=True)
    expected = source_digest(WORKSPACE / 'src')
    command = f"source {shlex.quote(str(ROS_SETUP))} && {shell_command(parts)}"
    subprocess.run(["/bin/bash", "-c", command], cwd=WORKSPACE, check=True)
    remember_built_sources(WORKSPACE / 'src', WORKSPACE / 'install', expected)


def prepare_ros(args: argparse.Namespace) -> None:
    ensure_ros()
    needs_build = args.build or not workspace_is_built()
    if args.dry_run:
        print(f"ROS build: {'yes' if needs_build else 'no'}")
        return
    if needs_build:
        build_workspace()
    if not WORKSPACE_SETUP.is_file():
        raise RuntimeError(f"ビルド済み環境が見つかりません: {WORKSPACE_SETUP}")


def exec_command(parts: list[str], cwd: Path = ROOT, dry_run: bool = False) -> int:
    print(f"command: {shell_command(parts)}", flush=True)
    if dry_run:
        return 0
    os.chdir(cwd)
    os.execvp(parts[0], parts)
    return 0


def exec_ros(parts: list[str], dry_run: bool = False) -> int:
    command = (
        f"source {shlex.quote(str(ROS_SETUP))} && "
        f"source {shlex.quote(str(WORKSPACE_SETUP))} && "
        f"exec {shell_command(parts)}"
    )
    print(f"command: {shell_command(parts)}", flush=True)
    if dry_run:
        return 0
    os.chdir(ROOT)
    os.execv("/bin/bash", ["/bin/bash", "-c", command])
    return 0


def system_launch_arguments(args: argparse.Namespace, action: str) -> list[str]:
    motion_mode = getattr(args, 'motion_mode', 'simultaneous')
    if args.no_tracker and motion_mode == 'staged_heading':
        raise ValueError('--motion-mode staged_heading requires the trajectory tracker')
    demo = action == "demo"
    motors = action == "full"
    parts = [
        "ros2",
        "launch",
        "omni_autonomy_next",
        "system.launch.py",
        f"demo:={'true' if demo else 'false'}",
        f"lidars:={'false' if demo else 'true'}",
        f"wheels:={'false' if demo else 'true'}",
        f"motors:={'true' if motors else 'false'}",
        f"gui:={'false' if args.no_gui else 'true'}",
        f"rviz:={'false' if args.no_rviz else 'true'}",
        f"navigation_delay:={args.navigation_delay}",
        f"initial_pose_id:={args.initial_pose_id}",
        # ここで渡さない限り、運用起動から MPPI へ戻す手段が無い。
        f"tracker:={'false' if args.no_tracker else 'true'}",
        f"motion_mode:={motion_mode}",
    ]
    # ROS 2 rejects an empty assignment (`goal_id:=`) as a malformed launch
    # argument.  Omitting it lets system.launch.py use its empty default and
    # leaves destination selection to the GUI.
    if args.goal_id:
        parts.append(f"goal_id:={args.goal_id}")
    return parts


def run_ros_system(args: argparse.Namespace, action: str) -> int:
    prepare_ros(args)
    labels = {"demo": "安全デモ", "real": "実機（motor OFF）", "full": "実機フル"}
    print(f"[start] {labels[action]}", flush=True)
    return exec_ros(system_launch_arguments(args, action), args.dry_run)


def run_ros_tool(args: argparse.Namespace, action: str) -> int:
    prepare_ros(args)
    if action == "gui":
        parts = ["ros2", "run", "omni_autonomy_next", "speed_gui"]
    elif action == "rviz":
        parts = ["rviz2", "-d", str(PACKAGE_SHARE / "rviz" / "system.rviz")]
    elif action == "check-odometry":
        parts = [
            sys.executable, str(ROOT / "scripts" / "check_odometry_signs.py"),
        ]
    elif action == "check-drive-directions":
        parts = [
            sys.executable,
            str(ROOT / "scripts" / "check_drive_directions.py"),
        ]
        if args.accept_motor_risk:
            parts.append("--accept-motor-risk")
    elif action == "diagnose-chain":
        parts = [
            sys.executable, str(ROOT / "scripts" / "diagnose_chain.py"),
            "--seconds", str(args.seconds),
        ]
    elif action == "record-run":
        parts = [sys.executable, str(ROOT / "scripts" / "record_run.py")]
    elif action == "goal":
        if not args.goal_id:
            raise RuntimeError("goal には --goal-id を指定してください")
        parts = [
            "ros2", "topic", "pub", "--once",
            "/navigation/goal_id_request", "std_msgs/msg/String",
            "{data: '" + args.goal_id + "'}",
        ]
    elif action in {"remember", "remembered-goal"}:
        if not args.pose_name.strip() and sys.stdin.isatty():
            args.pose_name = input("地点名 > ").strip()
        if not args.pose_name.strip():
            raise RuntimeError(f"{action} には --pose-name を指定してください")
        topic = (
            "/navigation/remember_pose_request"
            if action == "remember"
            else "/navigation/remembered_goal_request"
        )
        parts = [
            "ros2", "topic", "pub", "--once", topic,
            "std_msgs/msg/String",
            json.dumps({"data": args.pose_name.strip()}, ensure_ascii=False),
        ]
    else:
        if sys.stdin.isatty():
            if args.input_stl is None:
                value = input("入力STLのパス > ").strip()
                args.input_stl = Path(value) if value else None
            if args.output_yaml is None:
                value = input("出力YAMLのパス > ").strip()
                args.output_yaml = Path(value) if value else None
        if args.input_stl is None or args.output_yaml is None:
            raise RuntimeError(
                "cad-import には --input-stl と --output-yaml を指定してください"
            )
        parts = [
            "ros2", "run", "omni_autonomy_next", "cad_to_field",
            str(args.input_stl), str(args.output_yaml),
            "--slice-z-mm", str(args.slice_z_mm),
            "--simplify-mm", str(args.simplify_mm),
        ]
        if args.layout_config:
            parts.extend(["--layout-config", str(args.layout_config)])
    return exec_ros(parts, args.dry_run)


def build_gateway(dry_run: bool = False) -> None:
    require_program("cmake")
    configure = ["cmake", "-S", str(GATEWAY), "-B", str(GATEWAY_BUILD), "-DCMAKE_BUILD_TYPE=RelWithDebInfo"]
    build = ["cmake", "--build", str(GATEWAY_BUILD), "-j2"]
    if dry_run:
        print(f"configure: {shell_command(configure)}")
        print(f"build: {shell_command(build)}")
        return
    print("[setup] bacon6ゲートウェイをビルドします...", flush=True)
    expected = source_digest(GATEWAY)
    subprocess.run(configure, cwd=ROOT, check=True)
    subprocess.run(build, cwd=ROOT, check=True)
    remember_built_sources(GATEWAY, GATEWAY_BUILD, expected)


def run_gateway_action(args: argparse.Namespace, action: str) -> int:
    needs_build = (action == "gateway-build" or args.build or not GATEWAY_BINARY.is_file()
                   or not build_matches_sources(GATEWAY, GATEWAY_BUILD))
    if needs_build:
        build_gateway(args.dry_run)
    if action == "gateway-build":
        if not args.dry_run:
            print("[done] ゲートウェイのビルドが完了しました。")
        return 0
    if args.dry_run and not needs_build:
        print("gateway build: no")

    if action == "gateway-test":
        return exec_command(
            ["ctest", "--test-dir", str(GATEWAY_BUILD), "--output-on-failure"],
            dry_run=args.dry_run,
        )

    parts = [str(GATEWAY_BINARY)]
    if action == "gateway-monitor":
        parts.extend(["--jetson-only", "--dashboard"])
    elif args.dashboard:
        parts.append("--dashboard")
    return exec_command(parts, cwd=GATEWAY, dry_run=args.dry_run)


def run_action(args: argparse.Namespace, action: str) -> int:
    if action in ROS_SYSTEM_ACTIONS:
        return run_ros_system(args, action)
    if action in ROS_TOOL_ACTIONS:
        return run_ros_tool(args, action)
    if action == "ros-build":
        build_workspace(args.dry_run)
        if not args.dry_run:
            print("[done] ROS 2ワークスペースのビルドが完了しました。")
        return 0
    if action == "verify":
        return exec_command(["bash", str(ROOT / "scripts" / "verify_all.sh")], dry_run=args.dry_run)
    if action == "campaign":
        return exec_command(
            [
                sys.executable, "-m", "simulation.run_campaign",
                "--episodes", str(args.episodes),
                "--random-episodes", str(args.random_episodes),
                "--seed", str(args.seed),
            ],
            dry_run=args.dry_run,
        )
    if action == "autotune":
        return exec_command([sys.executable, "-m", "simulation.autotune"], dry_run=args.dry_run)
    if action == "rl-train":
        return exec_command(
            [
                sys.executable, "-m", "simulation.reinforcement_learning", "train",
                "--episodes", str(args.training_episodes),
                "--random-episodes", str(args.training_random_episodes),
                "--eval-episodes", str(args.eval_episodes),
                "--eval-random-episodes", str(args.eval_random_episodes),
                "--seed", str(args.seed),
                "--model", str(args.model),
                "--deploy-policy", str(
                    ROOT / "ros2_ws" / "src" / "omni_autonomy_next"
                    / "config" / "rl_policy.yaml"
                ),
            ],
            dry_run=args.dry_run,
        )
    if action == "rl-evaluate":
        return exec_command(
            [
                sys.executable, "-m", "simulation.reinforcement_learning", "evaluate",
                "--episodes", str(args.episodes),
                "--random-episodes", str(args.random_episodes),
                "--seed", str(args.seed),
                "--deployed-policy", str(
                    ROOT / "ros2_ws" / "src" / "omni_autonomy_next"
                    / "config" / "rl_policy.yaml"
                ),
            ],
            dry_run=args.dry_run,
        )
    if action in GATEWAY_ACTIONS:
        return run_gateway_action(args, action)
    raise RuntimeError(f"未対応の機能です: {action}")


def main() -> int:
    args = parse_args()
    action = resolve_action(args)
    if action is None:
        return 0
    return run_action(args, action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"エラー: 処理に失敗しました (終了コード {error.returncode})", file=sys.stderr)
        raise SystemExit(error.returncode) from error
    except RuntimeError as error:
        print(f"エラー: {error}", file=sys.stderr)
        raise SystemExit(1) from error
