## Validation Checklist

LLM outputs can look correct even when they are not. Use this checklist to quickly verify before trusting the result.


1. Review the logic
- Does the response follow the right steps for the task?
- Are the core operations correct?
- Is it using the right inputs and ignoring the wrong ones?
- Are the right types of data being treated the right way?
2. Validate the results
- Does the output make sense given what you know?
- Are there any obvious red flags (extreme values, empty results, missing pieces)?
- Are edge cases and missing data handled properly?
- Does the output actually answer what you asked?
3. Check consistency
- Does rerunning the same request give similar results?
- If results change, can you explain why?
4. Probe failure behavior
- Have you tried a small rewording to see if the output is stable?
- When it fails, does it fail clearly or return something misleading?
5. Refine and iterate
- Did you adjust your prompt after spotting issues?
- Does the new output pass checks 1 and 2?

