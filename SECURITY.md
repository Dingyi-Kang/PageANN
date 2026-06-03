# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in PageANN, please report it responsibly.

### How to Report

Please report security vulnerabilities by emailing:

**dingyikangosu@gmail.com**

Please include the following information in your report:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential security impact and affected components
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions of PageANN are affected
- **Proof of Concept**: If applicable, provide proof-of-concept code
- **Suggested Fix**: If you have ideas for fixing the issue

### Response Timeline

- You will receive an acknowledgment within **72 hours**
- We will provide a more detailed response within **7 days**
- We will work to address verified vulnerabilities as quickly as possible

### Disclosure Policy

- Please do **not** disclose the vulnerability publicly until a fix is released
- We will credit you for the discovery unless you prefer to remain anonymous
- We will coordinate the disclosure timeline with you

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| Older   | :x:                |

**Note**: We recommend always using the latest version of PageANN.

## Security Best Practices

When using PageANN:

1. **Input Validation**: Validate all input data before processing
2. **File Permissions**: Ensure proper file permissions for index files
3. **Resource Limits**: Set appropriate memory and I/O limits
4. **Network Security**: If exposing PageANN over network, use proper authentication
5. **Dependencies**: Keep all dependencies up to date

## Known Security Considerations

PageANN is primarily designed for trusted environments. Key considerations:

- **No built-in authentication**: PageANN does not include user authentication
- **File system access**: Requires read/write access to index files
- **Memory usage**: Can consume significant memory for large indices
- **DoS potential**: Malformed queries could cause resource exhaustion

If deploying in production or untrusted environments, implement appropriate security controls at the application layer.
