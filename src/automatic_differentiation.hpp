#pragma once

#include <immintrin.h>

#define _mm256_set_m128d(vh, vl) \
    _mm256_insertf128_pd(_mm256_castpd128_pd256(vl), (vh), 1)

template <typename T>
struct dual_t
{
    T a;
    T b;

    dual_t()
        : a(0), b(0)
    {
    }
    inline explicit dual_t(const T &value)
        : a(value), b(0)
    {
    }
    inline dual_t(const T &a, const T &b)
        : a(a), b(b)
    {
    }
    dual_t<T> &operator+=(const dual_t<T> &y)
    {
        *this = *this + y;
        return *this;
    }

    dual_t<T> &operator-=(const dual_t<T> &y)
    {
        *this = *this - y;
        return *this;
    }

    dual_t<T> &operator*=(const dual_t<T> &y)
    {
        *this = *this * y;
        return *this;
    }

    dual_t<T> &operator/=(const dual_t<T> &y)
    {
        *this = *this / y;
        return *this;
    }
    dual_t<T> &operator+=(const T &s)
    {
        *this = *this + s;
        return *this;
    }

    dual_t<T> &operator-=(const T &s)
    {
        *this = *this - s;
        return *this;
    }

    dual_t<T> &operator*=(const T &s)
    {
        *this = *this * s;
        return *this;
    }

    dual_t<T> &operator/=(const T &s)
    {
        *this = *this / s;
        return *this;
    }
};

template <typename T>
static inline dual_t<T> operator+(const dual_t<T> &f)
{
    return f;
}
template <typename T>
static inline dual_t<T> operator-(const dual_t<T> &f)
{
    return dual_t<T>(-f.a, -f.b);
}

template <typename T>
static inline dual_t<T> operator+(const dual_t<T> &f, const T &s)
{
    return dual_t<T>(f.a + s, f.b);
}
template <typename T>
static inline dual_t<T> operator+(const dual_t<T> &f, const dual_t<T> &g)
{
    return dual_t<T>(f.a + g.a, f.b + g.b);
}
template <typename T>
static inline dual_t<T> operator-(const dual_t<T> &f, const T &s)
{
    return dual_t<T>(f.a - s, f.b);
}
template <typename T>
static inline dual_t<T> operator-(const dual_t<T> &f, const dual_t<T> &g)
{
    return dual_t<T>(f.a - g.a, f.b - g.b);
}
template <typename T>
static inline dual_t<T> operator*(const dual_t<T> &f, const T &s)
{
    return dual_t<T>(f.a * s, f.b * s);
}
template <typename T>
static inline dual_t<T> operator*(const T &s, const dual_t<T> &f)
{
    return dual_t<T>(f.a * s, f.b * s);
}
template <typename T>
static inline dual_t<T> operator*(const dual_t<T> &f, const dual_t<T> &g)
{
    return dual_t<T>(f.a * g.a, f.b * g.a + f.a * g.b);
}
template <typename T>
static inline dual_t<T> operator/(const dual_t<T> &f, const dual_t<T> &g)
{
    return dual_t<T>(f.a / g.a, f.b / g.a - f.a * g.b / (g.a * g.a));
}
template <typename T>
static inline dual_t<T> operator/(const dual_t<T> &f, const T &s)
{
    return dual_t<T>(f.a / s, f.b / s);
}
template <typename T>
static inline bool operator>(const dual_t<T> &f, const dual_t<T> &g)
{
    return f.a > g.a;
}
template <typename T>
static inline dual_t<T> sqrt(const dual_t<T> &f)
{
    const auto df_0 = std::sqrt(f.a);
    const auto df_1 = T{1} / (T{2} * df_0);
    return dual_t<T>(
        df_0,
        f.b * df_1);
}
template <typename T>
static inline dual_t<T> sin(const dual_t<T> &f)
{
    const auto df_0 = std::sin(f.a);
    const auto df_1 = std::cos(f.a);
    return dual_t<T>(
        df_0,
        f.b * df_1);
}
template <typename T>
static inline dual_t<T> cos(const dual_t<T> &f)
{
    const auto df_0 = std::cos(f.a);
    const auto df_1 = -std::sin(f.a);
    return dual_t<T>(
        df_0,
        f.b * df_1);
}
template <typename T>
static inline dual_t<T> pow(const dual_t<T> &f, const T& g)
{
    const auto df_0 = std::pow(f.a, g);
    const auto df_1 = g * std::pow(f.a, g - T{1});
    return dual_t<T>(
        df_0,
        f.b * df_1);
}
template <typename T>
static inline dual_t<T> square(const dual_t<T> &f)
{
    const auto df_0 = f.a * f.a;
    const auto df_1 = T{2} * f.a;
    return dual_t<T>(
        df_0,
        f.b * df_1);
}

template <typename T>
struct hyper_dual_t
{
    T a;
    T b;
    T c;
    T d;

    hyper_dual_t()
        : a(0), b(0), c(0), d(0)
    {
    }
    hyper_dual_t(const hyper_dual_t<T> &other)
        : a(other.a), b(other.b), c(other.c), d(other.d)
    {
    }
    explicit hyper_dual_t(const T &value)
        : a(value), b(0), c(0), d(0)
    {
    }
    hyper_dual_t(const T &a, const T &b, const T &c, const T &d)
        : a(a), b(b), c(c), d(d)
    {
    }
    hyper_dual_t<T> &operator=(const hyper_dual_t<T> &other)
    {
        this->a = other.a;
        this->b = other.b;
        this->c = other.c;
        this->d = other.d;
        return *this;
    }
    hyper_dual_t<T> &operator+=(const hyper_dual_t<T> &y)
    {
        *this = *this + y;
        return *this;
    }

    hyper_dual_t<T> &operator-=(const hyper_dual_t<T> &y)
    {
        *this = *this - y;
        return *this;
    }

    hyper_dual_t<T> &operator*=(const hyper_dual_t<T> &y)
    {
        *this = *this * y;
        return *this;
    }

    hyper_dual_t<T> &operator/=(const hyper_dual_t<T> &y)
    {
        *this = *this / y;
        return *this;
    }
    hyper_dual_t<T> &operator+=(const T &s)
    {
        *this = *this + s;
        return *this;
    }

    hyper_dual_t<T> &operator-=(const T &s)
    {
        *this = *this - s;
        return *this;
    }

    hyper_dual_t<T> &operator*=(const T &s)
    {
        *this = *this * s;
        return *this;
    }

    hyper_dual_t<T> &operator/=(const T &s)
    {
        *this = *this / s;
        return *this;
    }
};

template <typename T>
static inline hyper_dual_t<T> operator+(const hyper_dual_t<T> &f)
{
    return f;
}
template <typename T>
static inline hyper_dual_t<T> operator-(const hyper_dual_t<T> &f)
{
    return hyper_dual_t<T>(-f.a, -f.b, -f.c, -f.d);
}

template <typename T>
static inline hyper_dual_t<T> operator+(const hyper_dual_t<T> &f, const hyper_dual_t<T> &g)
{
    return hyper_dual_t<T>(f.a + g.a, f.b + g.b, f.c + g.c, f.d + g.d);
}
template <typename T>
static inline hyper_dual_t<T> operator+(const hyper_dual_t<T> &f, const T &s)
{
    return f + hyper_dual_t<T>(s, 0, 0, 0);
}
template <typename T>
static inline hyper_dual_t<T> operator-(const hyper_dual_t<T> &f, const hyper_dual_t<T> &g)
{
    return hyper_dual_t<T>(f.a - g.a, f.b - g.b, f.c - g.c, f.d - g.d);
}
template <typename T>
static inline hyper_dual_t<T> operator-(const hyper_dual_t<T> &f, const T &s)
{
    return hyper_dual_t<T>(f.a - s, f.b, f.c, f.d);
}
template <typename T>
static inline hyper_dual_t<T> operator*(const hyper_dual_t<T> &f, const hyper_dual_t<T> &g)
{
    return hyper_dual_t<T>(
        f.a * g.a,
        f.a * g.b + f.b * g.a,
        f.a * g.c + f.c * g.a,
        f.a * g.d + f.b * g.c + f.c * g.b + f.d * g.a);
}
template <typename T>
static inline hyper_dual_t<T> operator*(const hyper_dual_t<T> &f, const T &s)
{
    return f * hyper_dual_t<T>(s, 0, 0, 0);
}
template <typename T>
static inline hyper_dual_t<T> operator*(const T &s, const hyper_dual_t<T> &f)
{
    return hyper_dual_t<T>(s, 0, 0, 0) * f;
}
template <typename T>
static inline hyper_dual_t<T> operator/(const hyper_dual_t<T> &f, const hyper_dual_t<T> &g)
{
    const auto inv = T{1} / g;
    return f * inv;
}
template <typename T>
static inline hyper_dual_t<T> operator/(const T &s, const hyper_dual_t<T> &f)
{
    const auto inv = hyper_dual_t<T>(
        T{1} / f.a,
        -f.b / (f.a * f.a),
        -f.c / (f.a * f.a),
        -f.d / (f.a * f.a) + T{2} * f.b * f.c / (f.a * f.a * f.a));
    return hyper_dual_t<T>(s * inv.a, s * inv.b, s * inv.c, s * inv.d);
}
template <typename T>
static inline hyper_dual_t<T> operator/(const hyper_dual_t<T> &f, const T &s)
{
    return f / hyper_dual_t<T>(s, 0, 0, 0);
}
template <typename T>
static inline bool operator>(const hyper_dual_t<T> &f, const hyper_dual_t<T> &g)
{
    return f.a > g.a;
}
template <typename T>
static inline hyper_dual_t<T> sqrt(const hyper_dual_t<T> &f)
{
    const auto df_0 = std::sqrt(f.a);
    const auto df_1 = T{1} / (T{2} * df_0);
    const auto df_2 = T{-2} * df_1 * df_1 * df_1;
    return hyper_dual_t<T>(
        df_0,
        f.b * df_1,
        f.c * df_1,
        f.d * df_1 + f.b * f.c * df_2);
}
template <typename T>
static inline hyper_dual_t<T> sin(const hyper_dual_t<T> &f)
{
    const auto df_0 = std::sin(f.a);
    const auto df_1 = std::cos(f.a);
    const auto df_2 = -df_0;
    return hyper_dual_t<T>(
        df_0,
        f.b * df_1,
        f.c * df_1,
        f.d * df_1 + f.b * f.c * df_2);
}
template <typename T>
static inline hyper_dual_t<T> cos(const hyper_dual_t<T> &f)
{
    const auto df_0 = std::cos(f.a);
    const auto df_1 = -std::sin(f.a);
    const auto df_2 = -df_0;
    return hyper_dual_t<T>(
        df_0,
        f.b * df_1,
        f.c * df_1,
        f.d * df_1 + f.b * f.c * df_2);
}

#include <immintrin.h>

template<typename T>
struct vec3_t
{
    T x;
    T y;
    T z;

    vec3_t()
        : x(), y(), z()
    {
    }
    vec3_t(T x, T y, T z)
        : x(x), y(y), z(z)
    {
    }
    inline T get_x() const
    {
        return x;
    }
    inline T get_y() const
    {
        return y;
    }
    inline T get_z() const
    {
        return z;
    }
};

template <typename T>
static inline vec3_t<T> operator-(const vec3_t<T> v)
{
    return vec3_t<T>(-v.x, -v.y, -v.z);
}
template <typename T>
static inline vec3_t<T> operator+(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return vec3_t<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
template <typename T>
static inline vec3_t<T> operator+(const vec3_t<T> v0, const T s)
{
    return vec3_t<T>(v0.x + s, v0.y + s, v0.z + s);
}
template <typename T>
static inline vec3_t<T> operator-(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return vec3_t<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
template <typename T>
static inline vec3_t<T> operator-(const vec3_t<T> v0, const T s)
{
    return vec3_t<T>(v0.x - s, v0.y - s, v0.z - s);
}
template <typename T>
static inline vec3_t<T> operator*(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return vec3_t<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
template <typename T>
static inline vec3_t<T> operator*(const vec3_t<T> v0, const T s)
{
    return vec3_t<T>(v0.x * s, v0.y * s, v0.z * s);
}
template <typename T>
static inline vec3_t<T> operator/(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return vec3_t<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
template <typename T>
static inline T dot(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
template <typename T>
static inline vec3_t<T> cross(const vec3_t<T> v0, const vec3_t<T> v1)
{
    return vec3_t<T>(
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x);
}
template <typename T>
static inline T l2norm(const vec3_t<T> &v)
{
    return dot(v, v);
}

template <>
struct vec3_t<float>
{
    union
    {
        __m128 v;
        struct
        {
            float x;
            float y;
            float z;
        };
    };

    vec3_t()
        : v(_mm_set1_ps(0))
    {
    }
    vec3_t(float x, float y, float z)
    {
        __m128 vx = _mm_load_ss(&x);
        __m128 vy = _mm_load_ss(&y);
        __m128 vz = _mm_load_ss(&z);
        __m128 vxy = _mm_unpacklo_ps(vx, vy);
        v = _mm_movelh_ps(vxy, vz);
    }
};

static inline vec3_t<float> operator-(const vec3_t<float> v)
{
    vec3_t<float> result;
    result.v = _mm_sub_ps(_mm_set1_ps(0.0f), v.v);
    return result;
}
static inline vec3_t<float> operator+(const vec3_t<float> v0, const vec3_t<float> v1)
{
    vec3_t<float> result;
    result.v = _mm_add_ps(v0.v, v1.v);
    return result;
}
static inline vec3_t<float> operator-(const vec3_t<float> v0, const vec3_t<float> v1)
{
    vec3_t<float> result;
    result.v = _mm_sub_ps(v0.v, v1.v);
    return result;
}
static inline vec3_t<float> operator*(const vec3_t<float> v0, const vec3_t<float> v1)
{
    vec3_t<float> result;
    result.v = _mm_mul_ps(v0.v, v1.v);
    return result;
}
static inline vec3_t<float> operator/(const vec3_t<float> v0, const vec3_t<float> v1)
{
    vec3_t<float> result;
    result.v = _mm_div_ps(v0.v, v1.v);
    return result;
}
static inline float dot(const vec3_t<float> v0, const vec3_t<float> v1)
{
    vec3_t<float> result;
    return _mm_cvtss_f32(_mm_dp_ps(v0.v, v1.v, 0x7f));
}

template <>
struct vec3_t<dual_t<float>>
{
    vec3_t<float> a;
    vec3_t<float> b;

    vec3_t()
        : a(), b()
    {
    }
    vec3_t(const dual_t<float> &x, const dual_t<float> &y, const dual_t<float> &z)
        : a(x.a, y.a, z.a), b(x.b, y.b, z.b)
    {
    }
    vec3_t(const vec3_t<float> &a, const vec3_t<float> &b)
        : a(a), b(b)
    {
    }
};

static inline vec3_t<dual_t<float>> operator+(const vec3_t<dual_t<float>> &f)
{
    return f;
}
static inline vec3_t<dual_t<float>> operator-(const vec3_t<dual_t<float>> &f)
{
    return vec3_t<dual_t<float>>(-f.a, -f.b);
}

static inline vec3_t<dual_t<float>> operator+(const vec3_t<dual_t<float>> &f, const vec3_t<float> &s)
{
    return vec3_t<dual_t<float>>(f.a + s, f.b);
}
static inline vec3_t<dual_t<float>> operator+(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
{
    return vec3_t<dual_t<float>>(f.a + g.a, f.b + g.b);
}
static inline vec3_t<dual_t<float>> operator-(const vec3_t<dual_t<float>> &f, const vec3_t<float> &s)
{
    return vec3_t<dual_t<float>>(f.a - s, f.b);
}
static inline vec3_t<dual_t<float>> operator-(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
{
    return vec3_t<dual_t<float>>(f.a - g.a, f.b - g.b);
}
static inline vec3_t<dual_t<float>> operator*(const vec3_t<dual_t<float>> &f, const vec3_t<float> &s)
{
    return vec3_t<dual_t<float>>(f.a * s, f.b * s);
}
static inline vec3_t<dual_t<float>> operator*(const vec3_t<float> &s, const vec3_t<dual_t<float>> &f)
{
    return vec3_t<dual_t<float>>(f.a * s, f.b * s);
}
static inline vec3_t<dual_t<float>> operator*(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
{
    return vec3_t<dual_t<float>>(f.a * g.a, f.b * g.a + f.a * g.b);
}
static inline vec3_t<dual_t<float>> operator/(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
{
    return vec3_t<dual_t<float>>(f.a / g.a, f.b / g.a - f.a * g.b / (g.a * g.a));
}
static inline vec3_t<dual_t<float>> operator/(const vec3_t<dual_t<float>> &f, const vec3_t<float> &s)
{
    return vec3_t<dual_t<float>>(f.a / s, f.b / s);
}
// static inline bool operator>(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
// {
//     return f.a > g.a;
// }

static inline dual_t<float> dot(const vec3_t<dual_t<float>> &f, const vec3_t<dual_t<float>> &g)
{
    return dual_t<float>(dot(f.a, g.a), dot(f.b, g.b));
}

template<>
struct vec3_t<double>
{
    union
    {
        __m256d v;
        struct
        {
            double x;
            double y;
            double z;
        };
    };

    vec3_t()
        : v(_mm256_set1_pd(0.0))
    {}
    inline vec3_t(double x, double y, double z)
    {
#if 0
        v = _mm256_set_pd(0.0, z, y, x);
#else
        __m128d vx = _mm_load_sd(&x);
        __m128d vy = _mm_load_sd(&y);
        __m128d vz = _mm_load_sd(&z);
        __m128d vxy = _mm_unpacklo_pd(vx, vy);
        v = _mm256_set_m128d(vz, vxy);
#endif
    }

    inline double get_x() const
    {
        return x;
    }
    inline double get_y() const
    {
        return y;
    }
    inline double get_z() const
    {
        return z;
    }
};

static inline vec3_t<double> operator-(const vec3_t<double> v)
{
    vec3_t<double> result;
    result.v = _mm256_sub_pd(_mm256_set1_pd(0.0), v.v);
    return result;
}
static inline vec3_t<double> operator+(const vec3_t<double> v0, const vec3_t<double> v1)
{
    vec3_t<double> result;
    result.v = _mm256_add_pd(v0.v, v1.v);
    return result;
}
static inline vec3_t<double> operator-(const vec3_t<double> v0, const vec3_t<double> v1)
{
    vec3_t<double> result;
    result.v = _mm256_sub_pd(v0.v, v1.v);
    return result;
}
static inline vec3_t<double> operator*(const vec3_t<double> v0, const vec3_t<double> v1)
{
    vec3_t<double> result;
    result.v = _mm256_mul_pd(v0.v, v1.v);
    return result;
}
static inline vec3_t<double> operator*(const vec3_t<double> v, const double s)
{
    vec3_t<double> result;
    result.v = _mm256_mul_pd(v.v, _mm256_set1_pd(s));
    return result;
}
static inline vec3_t<double> operator*(const double s, const vec3_t<double> v)
{
    vec3_t<double> result;
    result.v = _mm256_mul_pd(v.v, _mm256_set1_pd(s));
    return result;
}
static inline vec3_t<double> operator/(const vec3_t<double> v0, const vec3_t<double> v1)
{
    vec3_t<double> result;
    result.v = _mm256_div_pd(v0.v, v1.v);
    return result;
}
static inline double dot(const vec3_t<double> v0, const vec3_t<double> v1)
{
    __m256d xy = _mm256_mul_pd(v0.v, v1.v);
    __m256d temp = _mm256_hadd_pd(xy, xy);
    __m128d hi128 = _mm256_extractf128_pd(temp, 1);
    __m128d dotproduct = _mm_add_pd(_mm256_castpd256_pd128(temp), hi128);
    return _mm_cvtsd_f64(dotproduct);
}
static inline double l2norm(const vec3_t<double> &v)
{
    return dot(v, v);
}
static inline vec3_t<double> cross(const vec3_t<double> v0, const vec3_t<double> v1)
{
    // y1,z1,x1,w1
    __m256d vtemp1 = _mm256_permute4x64_pd(v0.v, _MM_SHUFFLE(3, 0, 2, 1));
    // z2,x2,y2,w2
    __m256d vtemp2 = _mm256_permute4x64_pd(v1.v, _MM_SHUFFLE(3, 1, 0, 2));
    // Perform the left operation
    __m256d vresult = _mm256_mul_pd(vtemp1, vtemp2);
    // z1,x1,y1,w1
    vtemp1 = _mm256_permute4x64_pd(vtemp1, _MM_SHUFFLE(3, 0, 2, 1));
    // y2,z2,x2,w2
    vtemp2 = _mm256_permute4x64_pd(vtemp2, _MM_SHUFFLE(3, 1, 0, 2));
    // Perform the right operation
    vtemp1 = _mm256_mul_pd(vtemp1, vtemp2);
    // Subract the right from left, and return answer
    vresult = _mm256_sub_pd(vresult, vtemp1);

    vec3_t<double> result;
    result.v = vresult;
    return result;
}
static inline double sum(const vec3_t<double> v)
{
    __m256d temp = _mm256_hadd_pd(v.v, v.v);
    __m128d hi128 = _mm256_extractf128_pd(temp, 1);
    __m128d dotproduct = _mm_add_pd(_mm256_castpd256_pd128(temp), hi128);
    return _mm_cvtsd_f64(dotproduct);
}

template <>
struct vec3_t<dual_t<double>>
{
    vec3_t<double> a;
    vec3_t<double> b;

    vec3_t()
        : a(), b()
    {
    }
    inline vec3_t(const dual_t<double> &x, const dual_t<double> &y, const dual_t<double> &z)
        : a(x.a, y.a, z.a), b(x.b, y.b, z.b)
    {
    }
    inline vec3_t(const vec3_t<double> &a, const vec3_t<double> &b)
        : a(a), b(b)
    {
    }
    inline explicit vec3_t(const vec3_t<double> &a)
        : a(a), b()
    {
    }

    inline dual_t<double> get_x() const
    {
        return dual_t<double>(a.x, b.x);
    }
    inline dual_t<double> get_y() const
    {
        return dual_t<double>(a.y, b.y);
    }
    inline dual_t<double> get_z() const
    {
        return dual_t<double>(a.z, b.z);
    }
};

static inline vec3_t<dual_t<double>> operator+(const vec3_t<dual_t<double>> &f)
{
    return f;
}
static inline vec3_t<dual_t<double>> operator-(const vec3_t<dual_t<double>> &f)
{
    return vec3_t<dual_t<double>>(-f.a, -f.b);
}

static inline vec3_t<dual_t<double>> operator+(const vec3_t<dual_t<double>> &f, const vec3_t<double> &s)
{
    return vec3_t<dual_t<double>>(f.a + s, f.b);
}
static inline vec3_t<dual_t<double>> operator+(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    return vec3_t<dual_t<double>>(f.a + g.a, f.b + g.b);
}
static inline vec3_t<dual_t<double>> operator-(const vec3_t<dual_t<double>> &f, const vec3_t<double> &s)
{
    return vec3_t<dual_t<double>>(f.a - s, f.b);
}
static inline vec3_t<dual_t<double>> operator-(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    return vec3_t<dual_t<double>>(f.a - g.a, f.b - g.b);
}
static inline vec3_t<dual_t<double>> operator*(const vec3_t<dual_t<double>> &f, const vec3_t<double> &s)
{
    return vec3_t<dual_t<double>>(f.a * s, f.b * s);
}
static inline vec3_t<dual_t<double>> operator*(const vec3_t<double> &s, const vec3_t<dual_t<double>> &f)
{
    return vec3_t<dual_t<double>>(f.a * s, f.b * s);
}
static inline vec3_t<dual_t<double>> operator*(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    return vec3_t<dual_t<double>>(f.a * g.a, f.b * g.a + f.a * g.b);
}
static inline vec3_t<dual_t<double>> operator*(const vec3_t<dual_t<double>> &v0, const dual_t<double> &s)
{
    return vec3_t<dual_t<double>>(v0.a * s.a, v0.a * s.b + v0.b * s.a);
}
static inline vec3_t<dual_t<double>> operator/(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    return vec3_t<dual_t<double>>(f.a / g.a, f.b / g.a - f.a * g.b / (g.a * g.a));
}
static inline vec3_t<dual_t<double>> operator/(const vec3_t<dual_t<double>> &f, const vec3_t<double> &s)
{
    return vec3_t<dual_t<double>>(f.a / s, f.b / s);
}
static inline dual_t<double> dot(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    const auto a = sum(f.a * g.a);
    const auto b = f.b * g.a + f.a * g.b;
    return dual_t<double>(a, sum(b));
}
static inline dual_t<double> l2norm(const vec3_t<dual_t<double>> &f)
{
    const auto f2_a = f.a * f.a;
    const auto f2_b = 2.0 * f.a * f.b;
    return dual_t<double>(sum(f2_a), sum(f2_b));
}

static inline vec3_t<dual_t<double>> cross(const vec3_t<dual_t<double>> &f, const vec3_t<dual_t<double>> &g)
{
    vec3_t<dual_t<double>> result;
    {
        // y1,z1,x1,w1
        __m256d vtemp1 = _mm256_permute4x64_pd(f.a.v, _MM_SHUFFLE(3, 0, 2, 1));
        // z2,x2,y2,w2
        __m256d vtemp2 = _mm256_permute4x64_pd(g.a.v, _MM_SHUFFLE(3, 1, 0, 2));
        // Perform the left operation
        __m256d vresult = _mm256_mul_pd(vtemp1, vtemp2);
        // z1,x1,y1,w1
        vtemp1 = _mm256_permute4x64_pd(vtemp1, _MM_SHUFFLE(3, 0, 2, 1));
        // y2,z2,x2,w2
        vtemp2 = _mm256_permute4x64_pd(vtemp2, _MM_SHUFFLE(3, 1, 0, 2));
        // Perform the right operation
        vtemp1 = _mm256_mul_pd(vtemp1, vtemp2);
        // Subract the right from left, and return answer
        result.a.v = _mm256_sub_pd(vresult, vtemp1);
    }
    {
        // y1,z1,x1,w1
        __m256d vtemp1 = _mm256_permute4x64_pd(f.a.v, _MM_SHUFFLE(3, 0, 2, 1));
        // z2,x2,y2,w2
        __m256d vtemp2 = _mm256_permute4x64_pd(g.a.v, _MM_SHUFFLE(3, 1, 0, 2));
        // y1,z1,x1,w1
        __m256d vtemp3 = _mm256_permute4x64_pd(f.b.v, _MM_SHUFFLE(3, 0, 2, 1));
        // z2,x2,y2,w2
        __m256d vtemp4 = _mm256_permute4x64_pd(g.b.v, _MM_SHUFFLE(3, 1, 0, 2));
        // Perform the left operation
        __m256d vresult = _mm256_add_pd(_mm256_mul_pd(vtemp1, vtemp4), _mm256_mul_pd(vtemp2, vtemp3));
        // y2,z2,x2,w2
        vtemp1 = _mm256_permute4x64_pd(vtemp1, _MM_SHUFFLE(3, 0, 2, 1));
        // z1,x1,y1,w1
        vtemp2 = _mm256_permute4x64_pd(vtemp2, _MM_SHUFFLE(3, 1, 0, 2));
        // y2,z2,x2,w2
        vtemp3 = _mm256_permute4x64_pd(vtemp3, _MM_SHUFFLE(3, 0, 2, 1));
        // z1,x1,y1,w1
        vtemp4 = _mm256_permute4x64_pd(vtemp4, _MM_SHUFFLE(3, 1, 0, 2));
        // Perform the right operation
        vtemp1 = _mm256_add_pd(_mm256_mul_pd(vtemp1, vtemp4), _mm256_mul_pd(vtemp2, vtemp3));
        // Subract the right from left, and return answer
        result.b.v = _mm256_sub_pd(vresult, vtemp1);
    }
    // Set w to zero
    return result;
}

template <typename T>
struct quat_t
{
    T x;
    T y;
    T z;
    T w;

    quat_t()
        : x(), y(), z(), w()
    {
    }
    quat_t(T w, T x, T y, T z)
        : x(x), y(y), z(z), w(w)
    {
    }
};

template <typename T>
static inline vec3_t<T> rotate(const quat_t<T>& q, const vec3_t<T> &v)
{
    const T t2 = q.w * q.x;
    const T t3 = q.w * q.y;
    const T t4 = q.w * q.z;
    const T t5 = -q.x * q.x;
    const T t6 = q.x * q.y;
    const T t7 = q.x * q.z;
    const T t8 = -q.y * q.y;
    const T t9 = q.y * q.z;
    const T t1 = -q.z * q.z;

    return vec3_t<T>(
        T(2) * ((t8 + t1) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
        T(2) * ((t4 + t6) * v.x + (t5 + t1) * v.y + (t9 - t2) * v.z) + v.y,
        T(2) * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z);
}

template <typename T>
static inline vec3_t<T> rotate_angle_axis(const vec3_t<T> &angle_axis, const vec3_t<T> &pt)
{
    const T theta2 = dot(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon()))
    {
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = T(1.0) / theta;

        const vec3_t<T> w = angle_axis * theta_inverse;

        const vec3_t<T> w_cross_pt = cross(w, pt);
        const T tmp = dot(w, pt) * (T(1.0) - costheta);

        return pt * costheta + w_cross_pt * sintheta + w * tmp;
    }
    else
    {
        const vec3_t<T> w_cross_pt = cross(angle_axis, pt);
        return pt + w_cross_pt;
    }
}

#if 1
static inline vec3_t<dual_t<double>> rotate(const quat_t<dual_t<double>>& q, const vec3_t<dual_t<double>>& v)
{
    const vec3_t<dual_t<double>> q_v(q.x, q.y, q.z);
    const auto uv = cross(q_v, v);
    const auto uuv = cross(q_v, uv);

    return v + ((uv * q.w) + uuv) * dual_t<double>(2);
}
#else
static inline vec3_t<dual_t<double>> rotate(const quat_t<dual_t<double>>& q, const vec3_t<dual_t<double>>& v)
{
    const auto t2 = q.w * q.x;
    const auto t3 = q.w * q.y;
    const auto t4 = q.w * q.z;
    const auto t5 = -q.x * q.x;
    const auto t6 = q.x * q.y;
    const auto t7 = q.x * q.z;
    const auto t8 = -q.y * q.y;
    const auto t9 = q.y * q.z;
    const auto t1 = -q.z * q.z;

    dual_t<double> x(v.a.x, v.b.x);
    dual_t<double> y(v.a.y, v.b.y);
    dual_t<double> z(v.a.z, v.b.z);

    return vec3_t<dual_t<double>>(
        dual_t<double>(2) * ((t8 + t1) * x + (t6 - t4) * y + (t3 + t7) * z) + x,
        dual_t<double>(2) * ((t4 + t6) * x + (t5 + t1) * y + (t9 - t2) * z) + y,
        dual_t<double>(2) * ((t7 - t3) * x + (t2 + t9) * y + (t5 + t8) * z) + z);
}
#endif
