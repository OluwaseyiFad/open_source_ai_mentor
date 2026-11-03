# Open Source Mentor AI

An assistant that helps developers understand GitHub issues, interpret unfamiliar codebases, and plan their first contributions.


## Sourcing Data

3 criteria matters most in choosing the right data for fine-tuning:
- Difficult 
- Quality
- Diversity
 
Bitterer lesson - Fine-tuning quality is bounded by data quality far more than by clever trainng optims.

Based on the 3 criteria defined above, below are the hard filters applied in selection of the issues:
- A descriptive issue body
- 1-8 files bounded PRs
- Not a bot, dependabot, docs-typo, or release PR
- Issues from repo with permissive licence
- Closed issues


I am going to use a retrieval and judge style meaning a model (Claude family) in my case will retrieve potential candidates of 200 such issues, and another model (Codex family) will judge.

Judge's job:
- Per-issue scoring: Is this a good example? Helps avoid position bias
- Set-level selection: Does the chosen set cover the space? 


Per-issue criteria is yet to be decided but must be sound - Everything else downstream is anchored on it.

Finally, I, the human in the loop will review these datasets to see if they deserve my approval.


Limitation Note: The steps outlines above in sourcing these dataset are not in any way indicative of the most optimal approach in sourcing dataset in production pipelines, however this is an experimental project and I am tinkering with approaches to see where it lands. 

