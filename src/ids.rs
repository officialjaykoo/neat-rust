use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;

macro_rules! define_numeric_id {
    ($name:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(i64);

        impl $name {
            pub const fn new(raw: i64) -> Self {
                Self(raw)
            }

            pub const fn raw(self) -> i64 {
                self.0
            }

            pub const fn next(self) -> Self {
                Self(self.0 + 1)
            }
        }

        impl From<i64> for $name {
            fn from(value: i64) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for i64 {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl Borrow<i64> for $name {
            fn borrow(&self) -> &i64 {
                &self.0
            }
        }

        impl PartialEq<i64> for $name {
            fn eq(&self, other: &i64) -> bool {
                self.0 == *other
            }
        }

        impl PartialOrd<i64> for $name {
            fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
                self.0.partial_cmp(other)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_numeric_id!(GenomeId);
define_numeric_id!(SpeciesId);
