mod binary;

#[derive(Clone, Debug)]
pub enum Operation {
    Constant,
    Variable,
    Placeholder,
    Add,
    Sub,
    Mul,
    Div,
}
