
# Find $U$

$$
\rho \xrightarrow[]{U} U \rho U^\dagger \xrightarrow[]{M} tr \left( M U \rho U^\dagger \right)
$$

First consider $A\otimes B$ 

$$
\begin{align}
tr \left( A \otimes B  \rho \right) & = \sum_{ j,k  }^{  } p \left( j,k  \right) a_{j} b_{k }  \\
& = \sum_{ j,k  }^{  } a_j b_k tr \left( \left| a_j  \right\rangle \left\langle a_j  \right| \otimes \left| a_k  \right\rangle \left\langle b_k  \right| \rho \right) \\
\end{align}
$$

Let $U_A \left| a_j  \right\rangle = \left| j  \right\rangle$ and $U_B \left| b_k  \right\rangle = \left| k \right\rangle$, then 

$$
\begin{align}
    tr \left( A \otimes B \rho  \right)
    & = \sum_{j, k   }^{  } a_j b_k tr \left( \left( U_A^\dagger \left| j \right\rangle \left\langle j \right| U_{A } \right) \otimes \left( U_{B }^{\dagger} \left| k \right\rangle \left\langle k \right| U_{B }  \right) \rho  \right) \\
    & = \sum_{ j, k  }^{  } a_j b_k tr \left( \left( U_A^{\dagger} \otimes U_{B }^{\dagger} \right) \left( \left| j \right\rangle \left\langle j \right| \otimes \left| k \right\rangle \left\langle k \right|  \right) \left( U_A \otimes U_B \right) \right) 
\end{align}
$$

Let $U = U_{A } \otimes U_{B }$ , $M = Z \otimes Z$, then 

$$
tr \left( A \otimes B \rho  \right) = tr \left( U^{\dagger} M U \rho  \right) = tr \left( M U \rho U^{\dagger} \right)
$$


Consider $A = \vec{a} \cdot \vec{\sigma} = \left| + \right\rangle \left\langle + \right| - \left| - \right\rangle \left\langle - \right|$, then $U = \left| 0 \right\rangle \left\langle + \right| + \left| 1 \right\rangle \left\langle - \right|$ . 

In $\Re^3$, $\vec{a} = \vec{r}(\theta, \phi)$, then 

$$
\begin{align}
    \left| + \right\rangle & = \vec{r}(\theta, \phi) = \cos{\frac{ \theta }{ 2  }} \left| 0 \right\rangle + e^{i \phi } \sin{ \frac{ \theta  }{ 2  }} \left| 1 \right\rangle \\
    \left| - \right\rangle & = \vec{r}(\pi - \theta, \pi + \phi) = \sin{\frac{ \theta  }{ 2  }} \left| 0 \right\rangle - e^{i \phi } \cos{ \frac{ \theta  }{ 2  }} \left| 1 \right\rangle
\end{align}
$$

We can calculate 

$$
\begin{align}
    U & = \left| 0 \right\rangle \left\langle + \right| + \left| 1 \right\rangle \left\langle - \right| \\
    & = \begin{bmatrix}
        \cos{ \frac{ \theta  }{ 2  }} & e^{-i\phi } \sin{ \frac{ \theta  }{ 2  }} \\
        \sin{\frac{ \theta  }{ 2 }} & -e^{-i\phi} \cos{\frac{ \theta  }{ 2 }}
    \end{bmatrix}
\end{align}
$$

For any $U$, using Z-Y decomposition 

$$
\begin{align}
    U & = e^{i \alpha} R_z(\beta) R_y(\gamma) R_z(\delta) \\ 
    & = e^{i \left( \alpha + \frac{ -\beta - \delta  }{ 2  } \right)}\begin{bmatrix}
        \cos{\frac{ \gamma  }{ 2  }} & -e^{i \delta } \sin{\frac{ \gamma  }{ 2  }} \\
        e^{i \beta}\sin{\frac{ \gamma  }{ 2 }} & e^{i (\beta + \delta)} \cos{\frac{ \gamma  }{ 2 }}
    \end{bmatrix}
\end{align}
$$

Let 
$$
\begin{cases}
    \gamma = \theta\\
    \beta = 0 \\
    \alpha = \frac{\delta  }{ 2  } = \frac{ \pi - \phi  }{ 2 } \\
    \delta = -\pi - \phi  = \pi - \phi
\end{cases}
$$

then 

$$
U_A = e^{i \left( \frac{ \pi  }{ 2  } - \frac{ \phi  }{ 2  } \right)} R_z(0) R_y(\theta ) R_z(\pi - \phi)
$$

