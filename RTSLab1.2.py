@@ -52,10 +52,15 @@ def correlation(t_arr, tau_arr, mx, number):
    print(f"Rxx execution time = {Rxx_time}")
    print(f"Rxу execution time = {Rxy_time}")

    if (Rxx_time > Rxy_time):
        print(f"Rxx exec time is higher than Rxy exec time")
    else:
        print(f"Rxy exec time is higher than Rxx exec time")

    plt.subplot(2, 1, 1)
    plt.ylabel("R(xx)")
    plt.plot([i for i in range(n // 2)], first_correlation)
    plt.subplot(2, 1, 2)
    plt.ylabel("R(xy)")
    plt.plot([i for i in range(n // 2)], second_correlation)
    plt.show() 
    plt.show()