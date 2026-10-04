# Optimization

The API's MILP accepts candidates with unique IDs, positions, clubs, prices, and caller-provided projected points. Binary variables select the 15-person 2/5/5/3 squad, 11 starters, one captain, and one distinct vice-captain. Constraints apply the £100m default budget, a maximum of three per club, and the standard 1 goalkeeper / 3–5 defenders / 2–5 midfielders / 1–3 forwards starting shape. The objective adds the captain's estimate a second time. Infeasible candidate sets return an error, never an invalid team.

This is selection optimization, not prediction. The service makes no claim about the origin or accuracy of caller-supplied projected points. Official 2026/27 FPL materials continue to state the 15-player 2/5/5/3 squad, £100m opening budget, and three-player club limit: [Premier League FPL squad basics](https://www.premierleague.com/en/news/2174419).

Transfer comparison is a single like-for-like alternative using only the supplied one-horizon estimates. It does not model hits, free transfers, price changes, chips, or future predicted value.
