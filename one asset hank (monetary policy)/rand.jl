using LinearAlgebra, Interpolations

# xs = [-1.0, 1.0, 5.0]
# interp = LinearInterpolation(xs, xs)
# interp(3) # == 3.0
# interp(-0.5) # == 0.5
# interp_extrap = LinearInterpolation(xs, xs, extrapolation_bc=Line()) # To exterpolate points outside of grid
# interp_extrap(-2)
# interp_extrap(10.0)


# The key to fixing knot vectors (a error that arises when xs has duplicate values) is to use dedeuplicate_knots. 
# Ex. xs = [-1.0, 1.0, 5.0, 5.0] will give an error when we run interp since there is a duplicate, dedeuplicate_knots adds a tiny ϵ to second value to make it slightly different and then re-orders knots
dub_xs = [-1.0, 5.0, 1.0, 5.0]
vals = [1,2,3,4]
fixed_xs = Interpolations.deduplicate_knots!(dub_xs; move_knots = true)
print(fixed_xs)
interp_extrap = LinearInterpolation(fixed_xs, vals, extrapolation_bc=Line()) # To exterpolate points outside of grid
interp_extrap(3)
