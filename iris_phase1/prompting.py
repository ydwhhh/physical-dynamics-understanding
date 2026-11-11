"""Prompt construction for the closed physical-dynamics label set."""

from __future__ import annotations


def classification_prompt(num_frames: int) -> str:
    noun = "image" if num_frames == 1 else "images"
    ordering = "" if num_frames == 1 else " The images are ordered chronologically."
    return (
        f"You are given {num_frames} {noun} sampled from one video.{ordering} "
        "Classify the physical dynamics shown into exactly one category:\n"
        "A. Dropping Ball\n"
        "B. Pendulum\n"
        "C. Sliding Cone\n"
        "D. Hitting Cones\n"
        "Reply with exactly one uppercase letter: A, B, C, or D. Do not add any other text."
    )
