pub(super) struct JsonObjectWriter<'a> {
    out: &'a mut String,
    needs_comma: bool,
}

impl<'a> JsonObjectWriter<'a> {
    pub(super) fn new(out: &'a mut String) -> Self {
        out.push('{');
        Self {
            out,
            needs_comma: false,
        }
    }

    pub(super) fn string_field(&mut self, key: &str, value: &str) {
        self.write_field_key(key);
        push_json_string(self.out, value);
    }

    pub(super) fn i64_field(&mut self, key: &str, value: i64) {
        self.write_field_key(key);
        self.out.push_str(&value.to_string());
    }

    pub(super) fn usize_field(&mut self, key: &str, value: usize) {
        self.write_field_key(key);
        self.out.push_str(&value.to_string());
    }

    pub(super) fn f64_field(&mut self, key: &str, value: f64) {
        self.write_field_key(key);
        if value.is_finite() {
            self.out.push_str(&value.to_string());
        } else {
            self.out.push_str("null");
        }
    }

    pub(super) fn bool_field(&mut self, key: &str, value: bool) {
        self.write_field_key(key);
        self.out.push_str(if value { "true" } else { "false" });
    }

    pub(super) fn i64_array_field(&mut self, key: &str, values: &[i64]) {
        self.write_field_key(key);
        push_json_i64_array(self.out, values);
    }

    pub(super) fn object_field(
        &mut self,
        key: &str,
        write_fields: impl FnOnce(&mut JsonObjectWriter<'_>),
    ) {
        self.write_field_key(key);
        let mut child = JsonObjectWriter::new(self.out);
        write_fields(&mut child);
        child.finish();
    }

    pub(super) fn raw_field(&mut self, key: &str, write_value: impl FnOnce(&mut String)) {
        self.write_field_key(key);
        write_value(self.out);
    }

    pub(super) fn finish(self) {
        self.out.push('}');
    }

    fn write_field_key(&mut self, key: &str) {
        if self.needs_comma {
            self.out.push(',');
        }
        self.needs_comma = true;
        push_json_string(self.out, key);
        self.out.push(':');
    }
}

fn push_json_i64_array(out: &mut String, values: &[i64]) {
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

pub(super) fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}
