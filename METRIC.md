# Defensive Dilemma Time (DDT)

DDT is a deterministic, pre-throw **geometric potential-stress** metric. It does not infer a defender's official responsibility, quarterback read, or causal effect because the regional 2021 data does not contain individual coverage assignments or defender-receiver matchup labels.

## Analysis population

A play is eligible only if the tracking file contains a `ball_snap`/`autoevent_ballsnap` frame followed by a `pass_forward`/`autoevent_passforward` frame. DDT uses frames from the snap through the frame immediately before the pass-forward event, recognizes WR/TE/RB players who PFF labels `Pass Route`, and evaluates defenders PFF labels `Coverage`.

## Coordinate and reachability definitions

All left-moving plays are rotated 180 degrees: `x_norm = 120 - x`, `y_norm = 53.3 - y`, and movement/orientation angles add 180 degrees modulo 360. Each route runner is projected 0.4 seconds using the observed speed and movement direction, while each defender's time to close to a 2-yard coverage cushion solves `distance = observed_speed × time + 0.5 × 4 yd/s² × time²`; a defender is geometrically relevant only if this estimated time is between 0.15 and 1.35 seconds for both receivers.

## Qualifying frame

A receiver pair qualifies only when both players are moving at least 1.5 yd/s, their projected separation is 3–18 yards, and the two projected routes form an angle of at least 50 degrees from a coverage defender. Severity combines the angular divergence and the defender's combined reach time, then reduces toward zero when another PFF coverage defender can reach either receiver within 0.75 seconds; a frame qualifies when severity is at least 0.25.

## Outputs and limits

`DDT = qualifying union frames × 0.1 seconds`, `DDT Rate = qualifying union frames / all snap-to-throw frames`, and `DSS` is mean qualifying severity weighted by the full pre-throw duration. Multiple simultaneous triples count once toward play-level DDT, while `Top triple DDT` reports only the single most frequently retained receiver A/receiver B/defender combination; it can therefore be lower than total DDT. The output can flag local geometric stress, but planned zone handoffs, bracket coverage, and support assignments remain possible explanations for any highlight.
