# Research Plan: Action Representation For UR5 + Wuji Block Placement

## Main question

For a vision-language-action policy controlling a UR5 and a 20-joint Wuji hand
to pick up a block and place it in a box, what do researchers and practitioners
report about Cartesian end-effector actions versus joint-position or
joint-velocity actions, and which representation best matches this system?

## Subtopics

1. **VLA and imitation-learning practice**
   - Inspect original OpenPI/pi0.5, DROID, Octo, OpenVLA, and related sources.
   - Identify action spaces actually used and the reasons authors give.

2. **Robot-control evidence**
   - Inspect original papers and official controller documentation comparing
     Cartesian commands with joint-space commands for manipulation.
   - Focus on inverse kinematics, singularities, safety, and transfer across poses.

3. **Practitioner experience**
   - Inspect first-party GitHub issues/discussions and direct practitioner reports.
   - Collect recurring benefits, failure modes, and caveats rather than popularity.

## Synthesis

Compare the evidence against this repository's actual interface: SpaceMouse
Cartesian teleoperation, UR5 `speedL`/`speedJ`, two RGB cameras, 6 measured UR5
joints, and 20 Wuji measured/commanded joints. Separate the recommended raw
recording schema from the policy action schema and deployment controller.
