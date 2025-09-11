# Development Approach Guidelines

## Core Principles

### Prototype First, Spec Later
- **Always start with a working prototype** (≤50 lines of code)
- **Test visually/functionally before adding complexity**
- **Write specs only after proving the approach works**
- **Avoid complex architectures until simple solutions are exhausted**

### Incremental Development
- **Make the smallest possible change that improves the current state**
- **Validate each change before proceeding to the next**
- **Keep each iteration independently testable and deployable**
- **Prefer 5 small working improvements over 1 large theoretical solution**

### Working Code Over Perfect Design
- **A 20-line script that solves 80% of the problem beats a 2000-line system that solves 100% theoretically**
- **Prioritize immediate usability over future extensibility**
- **Choose simple, obvious solutions over elegant, complex ones**
- **Refactor working code rather than designing perfect systems upfront**

## Anti-Patterns to Avoid

### Over-Engineering Red Flags
- **Creating multiple classes/modules before proving the concept works**
- **Writing comprehensive test suites for unvalidated approaches**
- **Building diagnostic systems before the core functionality works**
- **Designing for edge cases before handling the common case**

### Spec-Driven Development Problems
- **Writing detailed requirements without prototyping**
- **Creating complex architectures on paper**
- **Building abstractions before understanding concrete needs**
- **Focusing on completeness over functionality**

## Recommended Workflow

### For Algorithm Development
1. **Create a single-file prototype** (≤100 lines)
2. **Test with real data immediately**
3. **Validate output visually/functionally**
4. **Identify the top 3 remaining issues**
5. **Fix one issue at a time with minimal changes**
6. **Only refactor when the solution is proven to work**

### For Specs and Documentation
- **Write specs only after proving the approach works**
- **Keep requirements focused on immediate, validated needs**
- **Avoid designing for hypothetical future requirements**
- **Document what works, not what should work**

## Code Structure Guidelines

### Prefer Simple Structures
- **Single files over multiple modules until complexity demands it**
- **Functions over classes until state management is needed**
- **Direct algorithms over abstract frameworks**
- **Inline logic over extracted utilities until patterns emerge**

### When to Add Complexity
- **Only after the simple version works and is validated**
- **When the same pattern appears 3+ times**
- **When performance becomes a measurable problem**
- **When the simple approach cannot handle required use cases**

## Success Metrics

### For Individual Changes
- **Time to working prototype**: ≤2 hours
- **Lines of code for first working version**: ≤100
- **Time to visual validation**: ≤30 minutes after coding
- **Iterations to acceptable solution**: ≤5

### For Projects
- **Working solution exists at all times** (no broken intermediate states)
- **Each iteration improves the previous working version**
- **Complexity is added only when justified by validated needs**
- **Final solution is understandable by someone not involved in development**

This approach prioritizes **working solutions over perfect designs** and **incremental progress over comprehensive systems**.
