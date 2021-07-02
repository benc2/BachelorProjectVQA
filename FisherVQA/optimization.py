import numpy as np
import csv

def gradient_descent_backtrack_momentum(cost, gradient, start, mincost=-np.inf, alpha=0.5, c=0.5,
                                        maxstep=1, std_stop=None, maxiter=np.inf, momentum=0, log_filename=None,
                                        normalize=False):
    if log_filename is not None:
        log_file = open(log_filename, 'w')
        writer = csv.writer(log_file)
        # writer.writerow([3.14])
        # log_file.close()
        # exit()
    current_theta = start
    # print(current_theta)
    current_cost = cost(current_theta)
    cost_history = [current_cost]
    print("Starting:", current_cost, current_theta)
    if log_filename is not None:
        writer.writerow([0, current_cost] + list(current_theta))
    previous_change = 0
    i = 1
    while True:  # current_cost > maxcost:
        if i > maxiter:
            print("maxiter reached")
            break
        if current_cost < mincost:
            print(f"maxcost condition met: {mincost} > {current_cost}")
            break
        grad = gradient(current_theta)
        if normalize:
            grad /= np.linalg.norm(grad)
        print("grad")
        n = 0
        while True:
            change = alpha**n * (maxstep * grad + momentum*previous_change)
            updated_theta = current_theta - change
            updated_cost = cost(updated_theta)
            if updated_cost < current_cost - c * np.dot(grad, change):
                break
            n += 1
            print("*", end='')
            print("     ", updated_cost, updated_theta)
            if n>10:
                maxstep *= alpha
                break
        # if n>10:
        #     maxstep *= alpha**(n-10)
        print()
        previous_change = change
        current_theta = updated_theta
        # updated_cost = cost(current_theta)
        cost_history.append(updated_cost)
        current_cost = updated_cost
        if std_stop is not None:
            if np.std(cost_history[-6:]) < std_stop and len(cost_history) >= 5:
                print("Flatness condition met")
                return current_cost, current_theta

        print(i, round(updated_cost, 4), current_theta)
        if log_filename is not None:
            writer.writerow([i, updated_cost] + list(current_theta))

        i += 1


    if log_filename is not None:
        log_file.close()
    return current_cost, current_theta