# Research plan: real-robot multimodal synchronization

## Main question

How do major real-robot datasets and training pipelines handle asynchronous sensors,
dropped frames, timestamp alignment, and trajectory validation/filtering?

## Subtopics

1. DROID acquisition and post-processing
   - Camera/robot timestamp sources, alignment policy, dropped-sample behavior, and
     trajectory quality checks.
2. Open X-Embodiment / RT-X standardization
   - What RLDS standardization guarantees, what remains dataset-specific, and how
     invalid trajectories or missing modalities are represented or filtered.
3. Octo training data pipeline
   - How trajectories are transformed into fixed observation/action windows and how
     padding/masks prevent episode-boundary or missing-history contamination.

## Synthesis

Compare only mechanisms documented in primary sources (papers, official repositories,
and official documentation). Explicitly separate acquisition-time synchronization from
training-time resampling/windowing and call out where no universal guarantee exists.
