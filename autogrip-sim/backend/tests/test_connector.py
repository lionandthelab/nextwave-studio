"""Tests for the IsaacSimConnector and MockSimulator (app/sim_interface/connector.py)."""

from __future__ import annotations

import pytest

from app.sim_interface.connector import IsaacSimConnector, MockSimulator

@pytest.fixture()
def connector() -> IsaacSimConnector:
    """Provide a fresh IsaacSimConnector instance."""
    return IsaacSimConnector()


@pytest.fixture()
def mock_sim() -> MockSimulator:
    """Provide a fresh MockSimulator instance."""
    return MockSimulator()


# ---------------------------------------------------------------------------
# IsaacSimConnector tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartSimulation:
    """Tests for starting the simulation."""

    async def test_start_simulation(self, connector: IsaacSimConnector):
        """Starting a simulation should initialise the context."""
        result = await connector.start_simulation(headless=True)
        assert result is True
        assert connector.context is not None
        assert connector.context.running is True
        assert connector.context.headless is True
        assert len(connector.context.logs) >= 1

    async def test_start_simulation_gui_mode(self, connector: IsaacSimConnector):
        """Starting with headless=False should store the flag."""
        await connector.start_simulation(headless=False)
        assert connector.context is not None
        assert connector.context.headless is False


@pytest.mark.asyncio
class TestLoadScene:
    """Tests for loading the simulation scene."""

    async def test_load_scene_robot_object(self, connector: IsaacSimConnector):
        """Loading scene, robot, and object should populate context."""
        await connector.start_simulation()
        scene_ok = await connector.load_scene()
        robot_ok = await connector.load_robot("unitree_h1")
        obj_ok = await connector.load_object("/tmp/test.stl")

        assert scene_ok is True
        assert robot_ok is True
        assert obj_ok is True

        assert connector.context.ground_plane is True
        assert connector.context.robot is not None
        assert connector.context.robot.model == "unitree_h1"
        assert len(connector.context.objects) == 1

    async def test_load_scene_without_start_raises(
        self, connector: IsaacSimConnector
    ):
        """Loading scene before starting should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await connector.load_scene()

    async def test_load_robot_without_start_raises(
        self, connector: IsaacSimConnector
    ):
        """Loading robot before starting should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await connector.load_robot("test_bot")


@pytest.mark.asyncio
class TestExecuteCode:
    """Tests for executing code in the mock simulator."""

    async def test_execute_code_mock(self, connector: IsaacSimConnector):
        """Executing code should return a results dict with expected keys."""
        await connector.start_simulation()
        await connector.load_scene()
        await connector.load_robot("unitree_h1")
        await connector.load_object("/tmp/test.stl")

        code = 'torque = 5.0\ngrasp_width = 0.06\n'
        result = await connector.execute_code(code)

        assert isinstance(result, dict)
        assert "success" in result
        assert "duration" in result
        assert "logs" in result
        assert "frames" in result
        assert "object_final_state" in result
        assert "contact_forces" in result
        assert "joint_states" in result
        assert isinstance(result["duration"], float)
        assert result["duration"] > 0

    async def test_execute_code_without_start_raises(
        self, connector: IsaacSimConnector
    ):
        """Executing code before starting should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await connector.execute_code("pass")


@pytest.mark.asyncio
class TestSimulationLifecycle:
    """Tests for the full simulation lifecycle."""

    async def test_simulation_lifecycle(self, connector: IsaacSimConnector):
        """Full lifecycle: start -> load -> execute -> stop."""
        # Start
        await connector.start_simulation(headless=True)
        assert connector.context.running is True

        # Load
        await connector.load_scene()
        await connector.load_robot("franka")
        await connector.load_object("/tmp/cube.stl", position=(1.0, 0.0, 0.1))

        assert connector.context.robot.model == "franka"
        assert len(connector.context.objects) == 1

        # Execute
        code = 'torque = 8.0\ngrasp_width = 0.05\n'
        result = await connector.execute_code(code)
        assert "success" in result

        # Capture frames
        frames = await connector.capture_frames()
        assert isinstance(frames, list)

        # Stop
        await connector.stop_simulation()
        assert connector.context is None

    async def test_stop_without_start(self, connector: IsaacSimConnector):
        """Stopping without starting should not raise."""
        await connector.stop_simulation()  # Should be safe
        assert connector.context is None


# ---------------------------------------------------------------------------
# MockSimulator unit tests
# ---------------------------------------------------------------------------


class TestMockCodeQualityScoring:
    """Tests for MockSimulator code quality evaluation."""

    def test_code_with_high_torque(self, mock_sim: MockSimulator):
        """Code with explicit high torque should be reflected in quality."""
        code = "torque = 10.0\ngrasp_width = 0.06"
        quality = mock_sim.evaluate_code_quality(code)
        assert quality["torque_value"] == 10.0
        assert quality["grasp_width"] == 0.06

    def test_code_with_low_torque(self, mock_sim: MockSimulator):
        """Code with low torque should score lower."""
        code = "torque = 0.5\ngrasp_width = 0.15"
        quality = mock_sim.evaluate_code_quality(code)
        assert quality["torque_value"] == 0.5
        assert quality["grasp_width"] == 0.15

    def test_code_with_features(self, mock_sim: MockSimulator):
        """Code with error handling and contact checks should get bonuses."""
        code = """
torque = 5.0
try:
    contact_force = get_contact()
    hold_phase()
    sleep(5)
except Exception:
    pass
"""
        quality = mock_sim.evaluate_code_quality(code)
        assert quality["has_error_handling"] is True
        assert quality["has_contact_check"] is True
        assert quality["has_hold_phase"] is True

    def test_code_without_features(self, mock_sim: MockSimulator):
        """Minimal code should lack feature bonuses."""
        code = "pass"
        quality = mock_sim.evaluate_code_quality(code)
        assert quality["has_error_handling"] is False
        assert quality["has_contact_check"] is False
        assert quality["has_hold_phase"] is False

    def test_different_codes_different_probabilities(
        self, mock_sim: MockSimulator
    ):
        """Different code quality should produce different success probabilities."""
        bad_code = "pass"
        good_code = """
torque = 8.0
grasp_width = 0.06
try:
    contact = get_contact()
    hold_phase()
    sleep(5)
except:
    pass
"""
        bad_quality = mock_sim.evaluate_code_quality(bad_code)
        good_quality = mock_sim.evaluate_code_quality(good_code)

        bad_prob = mock_sim.compute_success_probability(bad_quality, iteration=1)
        good_prob = mock_sim.compute_success_probability(good_quality, iteration=1)

        assert good_prob > bad_prob


class TestMockFailureModes:
    """Tests for MockSimulator failure mode determination."""

    def test_low_torque_causes_slip(self, mock_sim: MockSimulator):
        """Low torque should bias toward slip failures."""
        quality = mock_sim.evaluate_code_quality("torque = 0.5")
        # Run multiple times to verify slip is possible
        modes = set()
        for _ in range(50):
            mode = mock_sim.determine_failure_mode(quality)
            modes.add(mode)
        assert "slip" in modes

    def test_wide_grasp_causes_no_contact(self, mock_sim: MockSimulator):
        """Overly wide grasp should bias toward no_contact."""
        quality = mock_sim.evaluate_code_quality("grasp_width = 0.20")
        modes = set()
        for _ in range(50):
            mode = mock_sim.determine_failure_mode(quality)
            modes.add(mode)
        assert "no_contact" in modes

    def test_narrow_grasp_causes_collision(self, mock_sim: MockSimulator):
        """Very narrow grasp width should bias toward collision."""
        code = "grasp_width = 0.01\napproach_height = 0.02"
        quality = mock_sim.evaluate_code_quality(code)
        modes = set()
        for _ in range(50):
            mode = mock_sim.determine_failure_mode(quality)
            modes.add(mode)
        assert "collision" in modes

    def test_failure_mode_always_returns_valid_string(
        self, mock_sim: MockSimulator
    ):
        """determine_failure_mode should always return a known failure type."""
        valid_modes = {"slip", "no_contact", "collision", "timeout"}
        quality = mock_sim.evaluate_code_quality("pass")
        for _ in range(100):
            mode = mock_sim.determine_failure_mode(quality)
            assert mode in valid_modes


class TestMockReset:
    """Tests for MockSimulator reset."""

    def test_reset_clears_iteration_count(self, mock_sim: MockSimulator):
        """Resetting the mock should zero the iteration counter."""
        mock_sim._iteration_count = 10
        mock_sim.reset()
        assert mock_sim._iteration_count == 0
