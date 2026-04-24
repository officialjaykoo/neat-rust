use std::collections::BTreeMap;

use super::ConfigError;

pub type IniDocument = BTreeMap<String, BTreeMap<String, String>>;

pub fn parse_ini(text: &str) -> Result<IniDocument, ConfigError> {
    let mut document: IniDocument = BTreeMap::new();
    let mut current_section: Option<String> = None;

    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let mut line = raw_line.trim();

        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }

        if line.starts_with('[') {
            let closing = line.find(']').ok_or_else(|| ConfigError::Parse {
                line: line_number,
                message: "missing closing ']'".to_string(),
            })?;
            let section = line[1..closing].trim();
            if section.is_empty() {
                return Err(ConfigError::Parse {
                    line: line_number,
                    message: "empty section name".to_string(),
                });
            }
            current_section = Some(section.to_string());
            document.entry(section.to_string()).or_default();
            line = line[closing + 1..].trim();
            if !line.is_empty() && !line.starts_with('#') && !line.starts_with(';') {
                return Err(ConfigError::Parse {
                    line: line_number,
                    message: "unexpected text after section header".to_string(),
                });
            }
            continue;
        }

        let section = current_section.clone().ok_or_else(|| ConfigError::Parse {
            line: line_number,
            message: "key/value found before any section".to_string(),
        })?;
        let Some(eq_index) = line.find('=') else {
            return Err(ConfigError::Parse {
                line: line_number,
                message: "expected key = value".to_string(),
            });
        };
        let key = line[..eq_index].trim();
        let value = strip_inline_comment(line[eq_index + 1..].trim()).trim();
        if key.is_empty() {
            return Err(ConfigError::Parse {
                line: line_number,
                message: "empty key".to_string(),
            });
        }
        document
            .entry(section)
            .or_default()
            .insert(key.to_string(), value.to_string());
    }

    Ok(document)
}

fn strip_inline_comment(value: &str) -> &str {
    let mut quote: Option<char> = None;
    for (idx, ch) in value.char_indices() {
        match ch {
            '\'' | '"' if quote == Some(ch) => quote = None,
            '\'' | '"' if quote.is_none() => quote = Some(ch),
            '#' | ';' if quote.is_none() => {
                let before = &value[..idx];
                if before.ends_with(char::is_whitespace) {
                    return before;
                }
            }
            _ => {}
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_ini_document() {
        let ini = parse_ini(
            r#"
            [NEAT]
            pop_size = 72
            key = value # comment
            quoted = "value # not comment"
            "#,
        )
        .expect("ini should parse");

        assert_eq!(ini["NEAT"]["pop_size"], "72");
        assert_eq!(ini["NEAT"]["key"], "value");
        assert_eq!(ini["NEAT"]["quoted"], "\"value # not comment\"");
    }

    #[test]
    fn rejects_key_before_section() {
        let err = parse_ini("pop_size = 72").expect_err("must reject orphan key");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }
}
