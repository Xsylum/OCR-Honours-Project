from label_generator import _get_verifier
read = _get_verifier()
p = "image_samples/Carabus_style_36.png"
print(repr(read(p)), read.nll(p, read(p).strip()), read.nll(p, "Carabus"), read.nll(p, "zzqxv"))