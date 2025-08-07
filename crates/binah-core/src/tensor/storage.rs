#[derive(Clone, Debug)]
pub enum TensorStorage {
    Bool { data: Vec<bool>, shape: Vec<usize> },

    U8 { data: Vec<u8>, shape: Vec<usize> },
    U16 { data: Vec<u16>, shape: Vec<usize> },
    U32 { data: Vec<u32>, shape: Vec<usize> },
    U64 { data: Vec<u64>, shape: Vec<usize> },
    U128 { data: Vec<u128>, shape: Vec<usize> },

    I8 { data: Vec<i8>, shape: Vec<usize> },
    I16 { data: Vec<i16>, shape: Vec<usize> },
    I32 { data: Vec<i32>, shape: Vec<usize> },
    I64 { data: Vec<i64>, shape: Vec<usize> },
    I128 { data: Vec<i128>, shape: Vec<usize> },

    F32 { data: Vec<f32>, shape: Vec<usize> },
    F64 { data: Vec<f64>, shape: Vec<usize> },
}

impl TensorStorage {
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorStorage::Bool { shape, .. } => shape,

            TensorStorage::U8 { shape, .. } => shape,
            TensorStorage::U16 { shape, .. } => shape,
            TensorStorage::U32 { shape, .. } => shape,
            TensorStorage::U64 { shape, .. } => shape,
            TensorStorage::U128 { shape, .. } => shape,

            TensorStorage::I8 { shape, .. } => shape,
            TensorStorage::I16 { shape, .. } => shape,
            TensorStorage::I32 { shape, .. } => shape,
            TensorStorage::I64 { shape, .. } => shape,
            TensorStorage::I128 { shape, .. } => shape,

            TensorStorage::F32 { shape, .. } => shape,
            TensorStorage::F64 { shape, .. } => shape,
        }
    }
}

pub trait IntoStorage: Clone + std::fmt::Debug + 'static {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage;
}

impl IntoStorage for bool {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::Bool { data, shape }
    }
}

impl IntoStorage for u8 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::U8 { data, shape }
    }
}

impl IntoStorage for u16 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::U16 { data, shape }
    }
}

impl IntoStorage for u32 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::U32 { data, shape }
    }
}

impl IntoStorage for u64 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::U64 { data, shape }
    }
}

impl IntoStorage for u128 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::U128 { data, shape }
    }
}

impl IntoStorage for i8 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::I8 { data, shape }
    }
}
impl IntoStorage for i16 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::I16 { data, shape }
    }
}
impl IntoStorage for i32 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::I32 { data, shape }
    }
}

impl IntoStorage for i64 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::I64 { data, shape }
    }
}

impl IntoStorage for i128 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::I128 { data, shape }
    }
}

impl IntoStorage for f32 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::F32 { data, shape }
    }
}

impl IntoStorage for f64 {
    fn into_storage(data: Vec<Self>, shape: Vec<usize>) -> TensorStorage {
        TensorStorage::F64 { data, shape }
    }
}
