# Contributing to Flock

Thank you for considering 
contributing to the Flock Framework! ❤️

We welcome contributions of all kinds, including
code, documentation and ideas. 

Please, make sure to read this document carefully 
before contributing to the Flock Framework.

This guide outlines the process and expectations for
making contributions to this project to help our
community of developers deliver the best
experience possible.

## Table of Content:
1. [Getting Started](#getting-started)
2. [Reporting Issues](#reporting-issues)
3. [Pull Requests](#pull-requests)
4. [Code Standards](#code-standards)
5. [Documentation Guidelines](#documentation-guidelines)
6. [Testing and Reliability](#testing-and-reliability)
7. [Code of Conduct](#code-of-conduct)
8. [License](#license)

## 🗒️ Getting Started:
1. **Fork the Repository**: Begin by forking the [main repository](https://github.com/whiteducksoftware/flock) to your own GitHub account.
2. **Clone the Fork**: Clone your fork to your local machine for development.
3. **Set Up Your Development Environment**: Follow the setup instructions in the [README](https://github.com/whiteducksoftware/flock/blob/master/README.md) to get the project up and running.


## ❗ Reporting Issues:
- Before submitting a new issue, pleas check [existing issues](https://github.com/whiteducksoftware/flock/issues) if it has already been reported.
- To submit a new issue, please use the provided **Issue Templates** and provide a clear and descriptive title along with a detailed description of the problem or feature request, including steps to reproduce if it's a bug.

## 🚋 Pull Requests: 
- Ensure your code is well-tested and adheres to the [Coding Standards](#code-standards) outlined below.
- Write clear commit messages that explain the changes made.
- Clearly outline and communicate breaking API-changes.
- Before submitting a pull request, make sure your branch is up to date with the base branch (`main`) of the [main repository](https://github.com/whiteducksoftware/flock).
- Open a pull request with a summary of your changes and any relevant issue numbers.

## 💻 Code Standards: 
- Flock follows a **declarative** approach and design philosophy. Make sure that your code follows this principle.
- Follow the coding conventions used within the existing codebase.
- Keep code modular and readable. Prefer clarity over brevity.
- Include comments where necessary and explain complex logic.

### A few best practices for writing good code:

#### Embrace Declarative Programming Principles:
- **Favor Declarative Styles**:
   - Exposed API-Code should express the desired results rather than detailing the control flow of the program.
   - Use declarative constructs to define behavior instead of imperative constructs where applicable.
   - Aim for expressiveness and clarity in stating "what" should happen rather than "how" it should happen.
- **Use High-Level Abstractions**:
   - Leverage Flock's existing abstractions to minimize boilerplate code and simplify the implementation.
   - This makes the code easier to read and understand.
#### Consistent Code Formatting:
- **Adhere to Existing Code Style**: Consistency in formatting enhances readability and helps developers navigate the codebase.
#### Meaningful Naming Conventions:
- **Use Descriptive Names**:
  - Choose variable, function, and class names that clearly describe their purpose.
  - Be consistent. Stick to the naming conventions in Flock's existing code-base.
#### Modular and Composable Code:
- **Write Modular Code**: Break down complex logic into smaller, reusable components.
- **Avoid Side Effects**: Aim to minimize side effects where possible.


## 📖 Documentation Guidelines: 
- Non-Source code documentation is to be provided in the [`docs`](https://github.com/whiteducksoftware/flock/tree/master/docs).
- Good documentation is crucial for the usability of Flock. When adding or updating code, please also update the relevant documentation.
- Use clear, concise language and include examples where applicable. (On that note: If you want to, you may also provide an example for the [example showcase](https://github.com/whiteducksoftware/flock-showcase)
- Maintain consistency in formatting and style throughout the documentation.

### A few best practices for writing good documentation:
1. Document the **Why**, not just the how:
    - Documentation should explain the rational behind your decisions, rather than just describing the "how".
    - This helps other developers understand the context and reasoning for the implementation, making the code more maintainable and modifyiable.
2. Keep it Up to Date:
   - Your documentation should evolve alongside the code. If you change the behavior of an existing component of Flock, please also take care to make sure the documentation reflects this fact.
3. Write for the Reader:
   - Consider the audience for your documentation.
   - It should be accessible and understandable for developers who are not intimately familiar with the code.
4. Document Intent and Design:
   - Refering to Point 1, Document your decisions of **why** you chose to implement a new component or code change the way you did, if it is not immediately obvious.
5. Code as Documentation:
   - Well-written code can serve as it's own documentation.
   - Code should be clear, expressive and self-explanatory where possible.
   - Using meaningful names for variables, functions, and classes can reduce the need for excessive comments.
6. Provide Examples:
   - Examples can help other developers understand your changes and are therefore encouraged.
7. Use Comments Wisely:
   - Aviod redundant comments that merely restate what the code does, which can clutter the codebase and detract from its readability.
8. Be Pragmatic:
   - There is no need for you to excessively comment every line of code you provide.
   - Add documentation where necessary and focus on keeping documentation on a high level.

## 🔭 Testing and Reliability:
Flock aims to provide a easy and reliable way to implement agentic applications. Therefore, well tested code is crucial.

- Test your changes thoroughly! Ensure that existing tests pass and add **new tests** for any new functionality.
- Follow Flock's testing conventions and use the provided testing framework.
- Run the tests before submitting your pull request to confirm that nothing is broken.

## Code of Conduct:

We are committed to creating a welcoming and inclusive environment for all contributors to Flock.
This Code of Conduct outlines our expectations for participants.

### Our Expectations:

1. **Be Respectful**: Treat all community members with kindness and respect, regardless of their background, identity, or experience level. We appreciate diverse perspectives and believe that collaboration is most effective in an inclusive atmosphere.
2. **Communicate Openly**: Promote open and constructive dialogue when discussing ideas, concerns, or feedback. Be supportive and considerate of others' opinions. Disagreements are natural and should be approached professionally and with empathy.
3. **Encourage Growth**: Offer constructive feedback that helps others grow, wether they are providing code contributions or engaging in discussions. Be supportive of newcomers and make an effort to help them understand the community norms.
4. **Civility Matters**: Maintain professionalism in all interactions. Harassment, bullying, or discrimination of any form will not be tolerated. Abuse of any kind-verbal or written-is unacceptable and may lead to removal from the community.
5. **Report Issues**: If you witness or experience behavior that violates our Code of Conduct, please report it to the project maintainers. We take all reports seriously and will address them promptly and fairly.

#### Consequences of Unacceptable Behavior:
Consequences for violating this Code of Conduct may include temporary or permanent expulsion from the community, depending on the severity of the behavior. We believe in the importance of accountability and will handle all reports with confidentiality and fairness.

#### Commitment to Improvement:
We encourage everyone to assist in creating an environment here where all contributors can thrive. Contious improvement is essential; therefore, we welcome feedback on our Code of Conduct (as well as our Code 😉) and our practices.


Thank you for contributing to Flock, the declarative Agent-Framework. 🦆💓


