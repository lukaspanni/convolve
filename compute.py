def convolve(image, output, kernel):
    channel_count = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    kernel_height = kernel.shape[0]
    kernel_halfh = kernel_height // 2
    kernel_width = kernel.shape[1]
    kernel_halfw = kernel_width // 2

    # Do convolution
    for x in range(image_width):
        for y in range(image_height):
            # Calculate usable image / kernel range
            x_min = max(0, x - kernel_halfw)
            x_max = min(image_width - 1, x + kernel_halfw)
            y_min = max(0, y - kernel_halfh)
            y_max = min(image_height - 1, y + kernel_halfh)

            # Convolve filter
            for c in range(channel_count):
                value = 0
                total = 0
                for u in range(x_min, x_max + 1):
                    for v in range(y_min, y_max + 1):
                        tmp = kernel[v - y + kernel_halfh, u - x + kernel_halfw]
                        value += image[c, v, u] * tmp
                        total += tmp
                output[c, y, x] = value / total
