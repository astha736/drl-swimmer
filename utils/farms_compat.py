"""Compatibility helpers for project code that used old FARMS conveniences."""

from __future__ import annotations

import numpy as np


def global_com_velocity(links, iteration: int) -> np.ndarray:
    """Mass-weighted center-of-mass velocity from FARMS link sensor data."""
    total_mass = 0.0
    mass_velocity = np.zeros(3)
    for link_i, mass in enumerate(links.masses):
        mass_velocity += mass * links.com_lin_velocity(
            iteration=iteration,
            link_i=link_i,
        )
        total_mass += mass
    assert total_mass > 0, "No masses"
    return mass_velocity / total_mass


def global_com_velocities(links) -> np.ndarray:
    """Mass-weighted center-of-mass velocity for all iterations."""
    velocities = np.asarray(links.com_lin_velocities())
    masses = np.asarray(links.masses, dtype=float)
    assert masses.size and np.sum(masses) > 0, "No masses"
    return np.sum(velocities * masses[None, :, None], axis=1) / np.sum(masses)


def joint_active_power(joints) -> np.ndarray:
    """Absolute active mechanical power per joint and timestep."""
    if hasattr(joints, "mechanical_power_active"):
        return np.abs(np.asarray(joints.mechanical_power_active()))
    return np.abs(np.asarray(joints.active_torques())) * np.abs(
        np.asarray(joints.velocities_all())
    )


def link_performance_metrics(links, timestep: float, n_iterations: int) -> dict:
    """Metrics formerly implemented on FARMS LinkSensorArray."""
    transient_steps = _transient_steps(timestep, n_iterations)
    positions = _global_com_positions(links)
    last = max(0, min(n_iterations, len(positions)) - 2)
    transient_start = min(transient_steps, last)

    path = _path_length(positions[: last + 1, :2], start=1)
    path_no_transient = _path_length(
        positions[: last + 1, :2],
        start=max(1, transient_start),
    )
    velocities = global_com_velocities(links)
    speed = np.linalg.norm(velocities[:last, :], axis=1)
    speed_no_transient = np.linalg.norm(velocities[transient_start:last, :], axis=1)

    return {
        "2_com_path_length_in_swimming_plane": path,
        "2_com_path_length_in_swimming_plane_noTransient": path_no_transient,
        "2_com_distance_travelled_start_end": float(
            np.linalg.norm(positions[last, :2] - positions[0, :2])
        ),
        "2_com_distance_travelled_start_end_noTransient": float(
            np.linalg.norm(positions[last, :2] - positions[transient_start, :2])
        ),
        "2_com_average_speed": float(np.mean(speed)) if speed.size else 0.0,
        "2_com_average_speed_noTransient": (
            float(np.mean(speed_no_transient)) if speed_no_transient.size else 0.0
        ),
        "2_link_0_path_length_in_swimming_plane (head)": _link_path_length(
            links,
            link_i=0,
            last=last,
        ),
        "2_link_10_path_length_in_swimming_plane (tail)": _link_path_length(
            links,
            link_i=min(10, links.size(1) - 1),
            last=last,
        ),
    }


def joint_performance_metrics(joints, timestep: float, n_iterations: int) -> dict:
    """Metrics formerly implemented on FARMS JointSensorArray."""
    transient_steps = _transient_steps(timestep, n_iterations)
    power = joint_active_power(joints)
    energy = power * timestep
    energy_no_transient = energy[transient_steps:]
    simulation_time = _conf_value("simulation_time_testing", timestep * n_iterations)
    return {
        "3_joints_total_energy (active torques)": float(np.sum(energy)),
        "3_joints_total_energy_noTransient (active torques)": float(
            np.sum(energy_no_transient)
        ),
        "3_joints_energy_per_second (active torques)": float(
            np.sum(energy) / simulation_time
        ),
    }


def _global_com_positions(links) -> np.ndarray:
    masses = np.asarray(links.masses, dtype=float)
    assert masses.size and np.sum(masses) > 0, "No masses"
    positions = np.asarray(links.com_positions())
    return np.sum(positions * masses[None, :, None], axis=1) / np.sum(masses)


def _link_path_length(links, link_i: int, last: int) -> float:
    positions = np.asarray(links.urdf_positions())[: last + 1, link_i, :2]
    return _path_length(positions, start=1)


def _path_length(positions: np.ndarray, start: int) -> float:
    if len(positions) <= start:
        return 0.0
    return float(
        np.sum(
            np.linalg.norm(
                positions[start:] - positions[start - 1 : -1],
                axis=1,
            )
        )
    )


def _transient_steps(timestep: float, n_iterations: int) -> int:
    transient = _conf_value("testing_transient", 0.0)
    steps = int(transient / timestep) if timestep else 0
    return min(max(steps, 0), max(0, n_iterations // 2))


def _conf_value(name: str, default):
    try:
        import conf
    except ModuleNotFoundError:
        return default
    return conf.CONF.get(name, default)
