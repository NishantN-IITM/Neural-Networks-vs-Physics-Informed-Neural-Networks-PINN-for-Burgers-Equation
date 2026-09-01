# Neural Network vs PINN for Burgers' Equation

## Learning from Sparse Measurements and Predicting Beyond the Data-Training Region

This project compares a conventional data-driven Neural Network (NN) with a Physics-Informed Neural Network (PINN) for the one-dimensional viscous Burgers' equation.

The governing equation is:

$$
u_t + u u_x = \nu u_{xx}
$$

with the computational domain:

$$
-1 \leq x \leq 1,
\qquad
0 \leq t \leq 1
$$

To focus on the comparison between the NN and PINN, reference data are generated using the exact travelling-wave solution.

$$
u(x,t)
=
\frac{1}{2}
-
\frac{1}{2}
\tanh\left(
\frac{x-\frac{1}{2}t}{4\nu}
\right)
$$


## Training Strategy

The conventional Neural Network is trained using only sparse solution measurements from the early-time region:


$$
0 \leq t \leq 0.35
$$


The PINN uses the same sparse measurements while additionally enforcing:

- Burgers' equation through the PDE residual
- The initial condition at \(t=0\)
- The boundary conditions at \(x=-1\) and \(x=1\)
- Physics constraints at collocation points distributed throughout the domain

### Prediction Beyond the Training Region

The region


$$
t > 0.35
$$


contains no supervised training data and is therefore considered the **data-training-excluded region**.

However, this region remains within the physical domain and is included in the PINN's physics-based training through the collocation points.

The trained NN and PINN are subsequently evaluated over the complete space-time domain and compared against the reference solution.


```python

import numpy as np                                      # numerical calculations
import matplotlib.pyplot as plt                         # plotting
import torch                                            # neural networks and automatic differentiation

np.random.seed(7)                                       # reproducible random points
torch.manual_seed(7)                                    # reproducible network initialization

device = "cpu"                                          # keep workshop code hardware-independent

nu = 0.03                                               # viscosity
x_min,x_max = -1.0,1.0                                  # spatial domain
t_min,t_max = 0.0,1.0                                   # time domain
t_data_max = 0.35                                       # data available only up to this time

```

## 1. Generate a reference solution

```python

def exact_solution(x,t):
    return 0.5-0.5*np.tanh((x-0.5*t)/(4*nu))             # exact travelling-wave solution


nx,nt = 121,121                                         # grid points used only for visualization
x_grid = np.linspace(x_min,x_max,nx)                    # spatial grid
t_grid = np.linspace(t_min,t_max,nt)                    # time grid
X,T = np.meshgrid(x_grid,t_grid)                        # space-time grid
U_exact = exact_solution(X,T)                           # reference solution

```

```python

plt.figure(figsize=(9,5))
plt.imshow(U_exact,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="u(x,t)")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("Reference solution of Burgers' equation")
plt.tight_layout()
plt.show()

```

<img width="838" height="490" alt="image" src="https://github.com/user-attachments/assets/c34821bb-05de-4a10-bf80-460a7e90be60" />

## 2. Pick sparse random measurements

```python
N_data = 100                                             # number of measured solution values

x_data = np.random.uniform(x_min,x_max,(N_data,1)).astype(np.float32)      # measured locations
t_data = np.random.uniform(t_min,t_data_max,(N_data,1)).astype(np.float32) # early-time measurements
u_data = exact_solution(x_data,t_data).astype(np.float32)                  # measured response

X_data = torch.tensor(np.hstack((x_data,t_data)),
                      dtype=torch.float32)                # network inputs [x,t]
U_data = torch.tensor(u_data,dtype=torch.float32)        # network targets u
```
```python
plt.figure(figsize=(9,5))
plt.imshow(U_exact,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="u(x,t)")
plt.scatter(x_data,t_data,facecolors="none",
            edgecolors="white",label="training data")
plt.axhline(t_data_max,linestyle="--",
            label="end of data-training region")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("Sparse random measurements used by both models")
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="838" height="490" alt="image" src="https://github.com/user-attachments/assets/b08ba4f9-442e-4e51-b98e-3c0fb928974e" />

## 3. Define the same neural-network architecture for both models

```python
def make_network():
    return torch.nn.Sequential(
        torch.nn.Linear(2,32),                          # inputs: x and t
        torch.nn.Tanh(),                                # smooth activation for derivatives
        torch.nn.Linear(32,32),                         # hidden layer
        torch.nn.Tanh(),                                # activation
        torch.nn.Linear(32,32),                         # hidden layer
        torch.nn.Tanh(),                                # activation
        torch.nn.Linear(32,1)                           # output: u(x,t)
    ).to(device)


model_nn = make_network()                               # ordinary data-driven neural network
model_pinn = make_network()                             # physics-informed neural network
```

## 4. Train the ordinary neural network

The ordinary neural network minimizes only the measurement error:

$$
\mathcal{L}_{\mathrm{NN}}
=
\frac{1}{N_d}
\sum_{i=1}^{N_d}
\left[
u_\theta(x_i,t_i)-u_i
\right]^2.
$$

```python
optimizer_nn = torch.optim.Adam(model_nn.parameters(),
                                lr=1e-3)                # Adam optimizer

epochs_nn = 1500                                       # number of NN training iterations
history_nn = []                                        # store loss values

for epoch in range(epochs_nn):
    u_pred = model_nn(X_data)                           # predict measured values
    loss_nn = torch.mean((u_pred-U_data)**2)            # data loss only

    optimizer_nn.zero_grad()                            # clear old gradients
    loss_nn.backward()                                  # backpropagation
    optimizer_nn.step()                                 # update weights and biases

    history_nn.append(loss_nn.item())                   # save loss value

    if epoch%500 == 0:
        print("NN epoch",epoch,"loss =",loss_nn.item())  # training progress
```
## 5. Prepare PINN points

```python
N_f = 500                                              # interior collocation points
N_ic = 80                                              # initial-condition points
N_bc = 80                                              # points on each boundary

x_f = np.random.uniform(x_min,x_max,(N_f,1)).astype(np.float32)           # collocation x
t_f = np.random.uniform(t_min,t_max,(N_f,1)).astype(np.float32)           # collocation t
X_f = torch.tensor(np.hstack((x_f,t_f)),
                   dtype=torch.float32,
                   requires_grad=True)                 # coordinates require gradients

x_ic = np.random.uniform(x_min,x_max,(N_ic,1)).astype(np.float32)         # initial x
t_ic = np.zeros((N_ic,1),dtype=np.float32)                                # t=0
u_ic = exact_solution(x_ic,t_ic).astype(np.float32)                        # initial condition
X_ic = torch.tensor(np.hstack((x_ic,t_ic)),dtype=torch.float32)            # IC inputs
U_ic = torch.tensor(u_ic,dtype=torch.float32)                              # IC targets

t_bc = np.random.uniform(t_min,t_max,(N_bc,1)).astype(np.float32)         # boundary times
x_left = x_min*np.ones((N_bc,1),dtype=np.float32)                          # x=-1
x_right = x_max*np.ones((N_bc,1),dtype=np.float32)                         # x=1

X_left = torch.tensor(np.hstack((x_left,t_bc)),dtype=torch.float32)        # left boundary
X_right = torch.tensor(np.hstack((x_right,t_bc)),dtype=torch.float32)      # right boundary

U_left = torch.tensor(exact_solution(x_left,t_bc).astype(np.float32),
                      dtype=torch.float32)                                  # left BC values
U_right = torch.tensor(exact_solution(x_right,t_bc).astype(np.float32),
                       dtype=torch.float32)                                 # right BC values



# Visualizing the Training points 

plt.figure(figsize=(8, 6))

# Interior collocation points
plt.scatter(
    x_f, t_f,
    s=12,
    label="Collocation points"
)

# Initial-condition points
plt.scatter(
    x_ic, t_ic,
    s=35,
    label="Initial condition"
)

# Left boundary
plt.scatter(
    x_left, t_bc,
    s=25,
    label="Boundary points"
)

# Right boundary
plt.scatter(
    x_right, t_bc,
    s=25
)

plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Training Points in the Space-Time Domain")
plt.legend()
plt.grid(True, linestyle="--")
plt.show()
```
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/0d459aeb-9f7b-4506-8aff-26ca9f11f5c7" />


## 6. Burgers' equation residual

For the PINN prediction \(u_\theta(x,t)\), automatic differentiation computes

$$
u_t,\qquad u_x,\qquad u_{xx}.
$$

The physics residual is

$$
r_\theta(x,t)
=
u_t+u_\theta u_x-\nu u_{xx}.
$$

```python
def burgers_residual(model,X):
    u = model(X)                                        # predicted solution

    grad_u = torch.autograd.grad(
        u,X,torch.ones_like(u),
        create_graph=True
    )[0]                                                # first derivatives

    u_x = grad_u[:,0:1]                                 # derivative with respect to x
    u_t = grad_u[:,1:2]                                 # derivative with respect to t

    grad_ux = torch.autograd.grad(
        u_x,X,torch.ones_like(u_x),
        create_graph=True
    )[0]                                                # derivatives of u_x

    u_xx = grad_ux[:,0:1]                               # second spatial derivative

    residual = u_t+u*u_x-nu*u_xx                       # Burgers' equation residual
    return residual
```


## 7. Train the PINN

The PINN minimizes

$$
\mathcal{L}_{\mathrm{PINN}}
=
10\mathcal{L}_{\mathrm{data}}
+
\mathcal{L}_{\mathrm{physics}}
+
10\mathcal{L}_{\mathrm{IC}}
+
10\mathcal{L}_{\mathrm{BC}}.
$$

```python

optimizer_pinn = torch.optim.Adam(model_pinn.parameters(),
                                  lr=1e-3)              # Adam optimizer

epochs_pinn = 2500                                     # number of PINN training iterations
history_pinn = []                                      # total PINN loss history

for epoch in range(epochs_pinn):
    loss_data = torch.mean((model_pinn(X_data)-U_data)**2)         # sparse data loss
    loss_physics = torch.mean(burgers_residual(model_pinn,X_f)**2) # PDE residual loss
    loss_ic = torch.mean((model_pinn(X_ic)-U_ic)**2)               # initial-condition loss

    loss_bc_left = torch.mean((model_pinn(X_left)-U_left)**2)      # left boundary
    loss_bc_right = torch.mean((model_pinn(X_right)-U_right)**2)   # right boundary
    loss_bc = loss_bc_left+loss_bc_right                           # total boundary loss

    loss_pinn = 10*loss_data+loss_physics+10*loss_ic+10*loss_bc   # total PINN loss

    optimizer_pinn.zero_grad()                          # clear old gradients
    loss_pinn.backward()                                # backpropagation
    optimizer_pinn.step()                               # update weights and biases

    history_pinn.append(loss_pinn.item())               # save total loss

    if epoch%500 == 0:
        print("PINN epoch",epoch,
              "total =",loss_pinn.item(),
              "physics =",loss_physics.item())          # training progress

```

## 8. Compare the training histories

```python

plt.figure(figsize=(8,4))
plt.semilogy(history_nn,label="ordinary NN")
plt.semilogy(history_pinn,label="PINN")
plt.xlabel("training epoch")                             # horizontal axis
plt.ylabel("loss")                                       # vertical axis
plt.title("Training histories")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

```
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/a7ecba97-51cb-48be-ab38-e0588afa97ad" />

## 9. Evaluate both models over the complete domain

```python

X_test = torch.tensor(
    np.column_stack((X.ravel(),T.ravel())).astype(np.float32),
    dtype=torch.float32
)                                                       # complete space-time query grid

with torch.no_grad():
    U_nn = model_nn(X_test).numpy().reshape(T.shape)     # ordinary NN prediction
    U_pinn = model_pinn(X_test).numpy().reshape(T.shape) # PINN prediction

error_nn = np.abs(U_nn-U_exact)                          # absolute NN error
error_pinn = np.abs(U_pinn-U_exact)                      # absolute PINN error

inside = T<=t_data_max                                   # inside data-training time range
outside = T>t_data_max                                   # outside data-training time range

rmse_nn_inside = np.sqrt(np.mean((U_nn[inside]-U_exact[inside])**2))
rmse_nn_outside = np.sqrt(np.mean((U_nn[outside]-U_exact[outside])**2))

rmse_pinn_inside = np.sqrt(np.mean((U_pinn[inside]-U_exact[inside])**2))
rmse_pinn_outside = np.sqrt(np.mean((U_pinn[outside]-U_exact[outside])**2))

print("RMSE inside data-training region")
print("Ordinary NN =",rmse_nn_inside)
print("PINN        =",rmse_pinn_inside)

print("\nRMSE outside data-training region")
print("Ordinary NN =",rmse_nn_outside)
print("PINN        =",rmse_pinn_outside)

```

## 10. Visualize the predicted fields

```python
plt.figure(figsize=(9,5))
plt.imshow(U_nn,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="predicted u(x,t)")
plt.axhline(t_data_max,linestyle="--")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("Ordinary neural-network prediction")
plt.tight_layout()
plt.show()
```
<img width="838" height="490" alt="image" src="https://github.com/user-attachments/assets/1ed62d43-ddad-45fa-b81b-63a72f11eb31" />

```python
plt.figure(figsize=(9,5))
plt.imshow(U_pinn,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="predicted u(x,t)")
plt.axhline(t_data_max,linestyle="--")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("PINN prediction")
plt.tight_layout()
plt.show()
```
<img width="838" height="490" alt="image" src="https://github.com/user-attachments/assets/32e1a889-e1e9-4597-92ee-d159686a6019" />

## 11. Visualize the errors

```python
plt.figure(figsize=(9,5))
plt.imshow(error_nn,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="absolute error")
plt.axhline(t_data_max,linestyle="--")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("Ordinary neural-network error")
plt.tight_layout()
plt.show()
```

<img width="847" height="490" alt="image" src="https://github.com/user-attachments/assets/e55328a0-cefd-49c0-986d-202746c56426" />



```python
plt.figure(figsize=(9,5))
plt.imshow(error_pinn,extent=[x_min,x_max,t_max,t_min],
           aspect="auto")
plt.colorbar(label="absolute error")
plt.axhline(t_data_max,linestyle="--")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("time t")                                     # vertical axis
plt.title("PINN error")
plt.tight_layout()
plt.show()
```
<img width="856" height="490" alt="image" src="https://github.com/user-attachments/assets/31b02ab9-a0a7-43bc-91dc-b1a537d4cff5" />

## 12. Compare interpolation and prediction outside the data-training region

```python
t_inside = 0.25                                         # time inside measurement region
index_inside = np.argmin(abs(t_grid-t_inside))          # nearest grid index

plt.figure(figsize=(9,4))
plt.plot(x_grid,U_exact[index_inside],label="exact")
plt.plot(x_grid,U_nn[index_inside],"--",label="ordinary NN")
plt.plot(x_grid,U_pinn[index_inside],label="PINN")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("u(x,t)")                                     # vertical axis
plt.title("Prediction inside data-training region: t=0.25")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
```

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/5aac6fca-ee2f-4842-aeda-a32da7c6b75b" />

```python
t_outside = 0.80                                        # time outside measurement region
index_outside = np.argmin(abs(t_grid-t_outside))        # nearest grid index

plt.figure(figsize=(9,4))
plt.plot(x_grid,U_exact[index_outside],label="exact")
plt.plot(x_grid,U_nn[index_outside],"--",label="ordinary NN")
plt.plot(x_grid,U_pinn[index_outside],label="PINN")
plt.xlabel("space x")                                    # horizontal axis
plt.ylabel("u(x,t)")                                     # vertical axis
plt.title("Prediction outside data-training region: t=0.80")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
```

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/43973710-7e54-4ba3-a837-4164da3f6498" />

## 13. Test random query points

```python
N_query = 300                                           # number of random query points

x_query = np.random.uniform(x_min,x_max,(N_query,1)).astype(np.float32) # random x
t_query = np.random.uniform(t_min,t_max,(N_query,1)).astype(np.float32) # random t
u_query = exact_solution(x_query,t_query).astype(np.float32)             # reference values

X_query = torch.tensor(np.hstack((x_query,t_query)),
                       dtype=torch.float32)               # query inputs

with torch.no_grad():
    u_query_nn = model_nn(X_query).numpy()               # ordinary NN predictions
    u_query_pinn = model_pinn(X_query).numpy()           # PINN predictions

query_outside = t_query[:,0]>t_data_max                  # late-time query points

query_rmse_nn = np.sqrt(
    np.mean((u_query_nn[query_outside]-u_query[query_outside])**2)
)                                                       # NN query error

query_rmse_pinn = np.sqrt(
    np.mean((u_query_pinn[query_outside]-u_query[query_outside])**2)
)                                                       # PINN query error

print("Random-query RMSE outside data-training region")
print("Ordinary NN =",query_rmse_nn)
print("PINN        =",query_rmse_pinn)
```
