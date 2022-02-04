# Design Idea

The design idea underlying `DataStructure` is to separate components of the code:

- The first component may participate automatic differentiation;

- The second component never participates automatic differentiation.

The data structures defined in `DataStructure` should never hold data in the first component, and should incoporate data in the second component as much as possible. 